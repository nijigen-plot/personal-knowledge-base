import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { McpAgent } from 'agents/mcp';
import { z } from 'zod';

export class MyMCP extends McpAgent {
	server = new McpServer({
		name: 'MyMCP Server',
		version: '0.1.0',
	});

	async init() {
		this.server.tool(
			// ツールの名前
			'dice_roll',
			// ツールの説明
			'サイコロを降った結果を返します',
			// ツールの引数のスキーマ
			{ sides: z.number().min(1).max(100).default(6).describe('サイコロの面の数') },
			// ツールの実行関数
			async ({ sides }) => {
				// サイコロを振る
				const result = Math.floor(Math.random() * sides) + 1;
				// 結果を返す
				return {
					content: [{ type: 'text', text: result.toString() }],
				};
			}
		);
	}
}
