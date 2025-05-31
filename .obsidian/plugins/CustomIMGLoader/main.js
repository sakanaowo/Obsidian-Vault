import { Plugin, TFile, Notice } from "obsidian";

export default class ImagePasteLogger extends Plugin {
    async onload() {
        this.registerEvent(
            this.app.vault.on("create", async (file: TFile) => {
                if (file.extension.match(/png|jpg|jpeg|gif/)) {
                    const path = file.path;
                    const filename = file.name;

                    // You can show this in a Notice or console
                    new Notice(`📸 Image pasted: ${filename}\n📁 Location: ${path}`);
                    console.log(`[ImagePasteLogger] Image pasted:\n- Name: ${filename}\n- Path: ${path}`);
                }
            })
        );
    }
}
