const fs = require('fs-extra');
const path = require('path');
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const ignore = require('ignore');
const ig = ignore().add(['node_modules', '.git', 'dist', 'build', '.env', 'package-lock.json']);

async function getFiles(dir) {
    let results = [];
    const list = await fs.readdir(dir);

    for (const file of list) {
        const filePath = path.join(dir, file);
        const stat = await fs.stat(filePath);

        const relativePath = path.relative(process.cwd(), filePath);
        if (ig.ignores(relativePath)) continue;

        if (stat.isDirectory()) {
            results = results.concat(await getFiles(filePath));
        } else if (['.js', '.ts', '.py', '.cpp', '.sql'].includes(path.extname(file))) {
            results.push(filePath);
        }
    }
    return results;
}

async function startIngestion(targetPath) {
    console.log(`🔍 Scanning: ${targetPath}...`);
    const files = await getFiles(targetPath);
    
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 100,
    });

    const allDocs = [];

    for (const file of files) {
        const content = await fs.readFile(file, 'utf8');
        // Create LangChain Documents with source metadata
        const docs = await splitter.createDocuments(
            [content], 
            [{ source: path.basename(file) }]
        );
        allDocs.push(...docs);
    }

    console.log(`✅ Processed ${files.length} files into ${allDocs.length} chunks.`);
    return allDocs;
}

module.exports = { startIngestion };
