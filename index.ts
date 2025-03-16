import { PrismaClient } from "@prisma/client";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import fetch from "node-fetch";
import fs from "node:fs/promises";

const prisma = new PrismaClient();
const documentSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 512,
    chunkOverlap: 100,
});

const saveChunk = async (chunk: string) => {
    try {
        const embedding = await generateEmbedding(chunk);
        const embeddingFormat = `[${embedding.join(",")}]`;
        await prisma.$executeRaw`INSERT INTO "Embeddings" (id, value, vector)VALUES (default, ${chunk}, ${embeddingFormat}::vector)`;
    } catch (e) {
        throw new Error("Ошибка при сохранении фрагментов");
    }
};

const generateEmbedding = async (chunk: string) => {
    try {
        const response = await fetch("http://localhost:8000/embedding", {
            method: "POST",
            body: JSON.stringify({ chunks: [chunk] }),
            headers: { "Content-Type": "application/json" },
        });

        const json = (await response.json()) as Record<string, number[][]>;
        return json.embeddings[0];
    } catch (e) {
        throw new Error("Generate embedding error");
    }
};

const saveDocument = async () => {
    const document = await fs.readFile("document.txt", "utf-8");
    const chunks = await documentSplitter.splitText(document);

    for (const chunk of chunks) {
        await saveChunk(chunk);
    }

    console.log("Done");
};

interface QwenRequest {
    prompt: string;
    system: string;
}

const generateQwen = async (request: QwenRequest) => {
    try {
        const response = await fetch("http://localhost:8001/generate", {
            method: "POST",
            body: JSON.stringify(request),
            headers: { "Content-Type": "application/json" },
        });

        const json = (await response.json()) as { response: string };
        return json.response;
    } catch (e) {
        throw new Error("Generate qwen error");
    }
};

const kNN = async (chunk: string, k: number) => {
    try {
        const embedding = await generateEmbedding(chunk);
        const embeddingFormat = `[${embedding.join(",")}]`;
        const similarChunks =
            await prisma.$queryRaw`SELECT id, value FROM "Embeddings" ORDER BY vector <-> ${embeddingFormat}::vector LIMIT ${k}`;

        return similarChunks as { id: number; value: string }[];
    } catch (e) {
        throw new Error("Ошибка при получении ближайших соседей");
    }
};

async function main() {
    const request = {
        system: "Отвечай кратко, опираясь на контекст",
        prompt: "Что болело у Пашки в рассказе Чехова Беглец?",
    };

    const context = await kNN(request.prompt, 3);
    // request.prompt += ` Контекст: ${context.map((c) => c.value).join(",")}`;

    const response = await generateQwen(request);
    console.log(request);
    console.log(response);
}

main()
    .then(async () => {
        await prisma.$disconnect();
    })
    .catch(async (e) => {
        console.error(e);
        await prisma.$disconnect();
        process.exit(1);
    });
