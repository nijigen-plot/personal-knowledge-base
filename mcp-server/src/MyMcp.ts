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
			'quarkgabber-knowledge-base-search',
			// ツールの説明
			'QuarkgabberナレッジベースAPI searchにアクセスしレスポンスを受け取ります。',
			// ツールの引数のスキーマ
            { query: z.string().describe('検索クエリ')},
			// ツールの実行関数
            async (args: any) => {
                const { query } = args;
                const response = await fetch('https://home.quark-hardcore.com/personal-knowledge-base/api/v1/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        k: 10
                    })
                });

                const data = await response.json();

                return {
                    content: [{ type: 'text', text: JSON.stringify(data, null, 2)}]
                };
        });
    }
}
