import { Anthropic } from "@anthropic-ai/sdk"
import { AnthropicVertex } from "@anthropic-ai/vertex-sdk"
import { VertexAI, GenerateContentStreamResult } from "@google-cloud/vertexai"
import { ApiHandler } from "../"
import { ApiHandlerOptions, ModelInfo, vertexDefaultModelId, VertexModelId, vertexModels } from "../../shared/api"
import { ApiStream } from "../transform/stream"

// https://docs.anthropic.com/en/api/claude-on-vertex-ai
export class VertexHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private anthropicClient: AnthropicVertex
	private geminiClient: VertexAI

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.anthropicClient = new AnthropicVertex({
			projectId: this.options.vertexProjectId,
			// https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#regions
			region: this.options.vertexRegion,
		})
		this.geminiClient = new VertexAI({
			project: this.options.vertexProjectId!,
			location: this.options.vertexRegion!,
		})
	}

	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const model = this.getModel()

		if (model.id.startsWith("gemini")) {
			const geminiModel = await this.geminiClient.getGenerativeModel({ model: model.id })
			try {
				const stream = await geminiModel.generateContentStream({
					contents: [{ role: 'user', parts: [{ text: messages[messages.length - 1].content }] }],
					safetySettings: [
						{
							category: "HARM_CATEGORY_DANGEROUS",
							threshold: "BLOCK_NONE",
						},
					],
				})

				for await (const chunk of stream.stream) {
					if (chunk.candidates?.[0]?.content?.parts?.[0]?.text) {
						yield {
							type: "text",
							text: chunk.candidates[0].content.parts[0].text,
						}
					}
				}
			} catch (error) {
				throw new Error(`Gemini API error: ${(error as Error).message}`)
			}
		} else {
			// Existing Claude handling code
			const stream = await this.anthropicClient.messages.create({
				model: model.id,
				max_tokens: model.info.maxTokens || 8192,
				temperature: 0,
				system: systemPrompt,
				messages,
				stream: true,
			})
			for await (const chunk of stream) {
				switch (chunk.type) {
					case "message_start":
						const usage = chunk.message.usage
						yield {
							type: "usage",
							inputTokens: usage.input_tokens || 0,
							outputTokens: usage.output_tokens || 0,
						}
						break
					case "message_delta":
						yield {
							type: "usage",
							inputTokens: 0,
							outputTokens: chunk.usage.output_tokens || 0,
						}
						break

					case "content_block_start":
						switch (chunk.content_block.type) {
							case "text":
								if (chunk.index > 0) {
									yield {
										type: "text",
										text: "\n",
									}
								}
								yield {
									type: "text",
									text: chunk.content_block.text,
								}
								break
						}
						break
					case "content_block_delta":
						switch (chunk.delta.type) {
							case "text_delta":
								yield {
									type: "text",
									text: chunk.delta.text,
								}
								break
						}
						break
				}
			}
		}
	}

	getModel(): { id: VertexModelId; info: ModelInfo } {
		const modelId = this.options.apiModelId
		if (modelId && modelId in vertexModels) {
			const id = modelId as VertexModelId
			return { id, info: vertexModels[id] }
		}
		return { id: vertexDefaultModelId, info: vertexModels[vertexDefaultModelId] }
	}
}
