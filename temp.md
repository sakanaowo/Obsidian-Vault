## Skill Set: AI-Assisted Learning với Obsidian

Đây là một bộ skill khá phức tạp vì nó kết hợp **pedagogy** (học thuật), **PKM** (Personal Knowledge Management), và **AI workflows**. Để xây dựng tốt, cần chia thành các skill module sau:

---

## 🧩 Các Skill Module Cần Có

### 1. `knowledge-intake` — Tiếp nhận & phân tích kiến thức

Claude/Cursor cần biết cách:

- Phân tích một tài liệu/topic → xác định **độ khó**, **prerequisite**, **core concepts**
- Phân loại kiến thức: declarative vs procedural vs conceptual
- Tóm tắt theo nhiều mức độ (ELI5 → Expert)

### 2. `note-structure` — Cấu trúc ghi chú cho Obsidian

Quy ước cụ thể mà skill phải encode:

- **Atomic note format**: 1 note = 1 concept (Zettelkasten principle)
- **Frontmatter schema**: `tags`, `status`, `source`, `created`, `related`
- **Link strategy**: `[[wikilinks]]` đến concepts liên quan
- **Folder structure**: `00-Inbox/`, `10-Literature/`, `20-Permanent/`, `30-Projects/`
- Template cho từng loại note: Concept Note, Literature Note, Evergreen Note, MOC (Map of Content)

### 3. `spaced-repetition-integration` — Kết hợp ôn tập

- Tạo flashcard từ note (tương thích với plugin **Obsidian Spaced Repetition** hoặc Anki)
- Format câu hỏi: cloze deletion, Q&A, image occlusion description
- Tag `#review/hard`, `#review/medium` tự động theo nội dung

### 4. `learning-path-generator` — Tạo lộ trình học

- Từ một topic → sinh ra **dependency graph** các khái niệm
- Đề xuất thứ tự học, ước lượng thời gian
- Output dưới dạng Obsidian Canvas hoặc MOC note

### 5. `progressive-summarization` — Kỹ thuật của Tiago Forte

- Layer 1: Raw capture
- Layer 2: Bold key passages
- Layer 3: Highlight the best
- Layer 4: Mini-summary
- Skill này cần Claude hiểu đang ở layer nào và làm gì tiếp

### 6. `connection-finder` — Tìm liên kết giữa concepts

- Đọc vault context → gợi ý `[[backlinks]]` phù hợp
- Phát hiện **knowledge gaps** trong vault hiện tại
- Gợi ý merge notes trùng lặp

### 7. `obsidian-plugin-awareness` — Hiểu ecosystem plugin

Skill phải biết syntax/behavior của các plugin phổ biến:

- **Dataview**: query language để tạo dynamic tables
- **Templater**: JS templating
- **Tasks**: `- [ ] task 📅 2024-01-01`
- **Excalidraw**: embed diagram
- **Kanban**, **Calendar**, **QuickAdd**

---

## 📚 Nguồn Tìm Để Xây Skill

### Về PKM & Obsidian methodology

| Nguồn                                                         | Nội dung                               |
| ------------------------------------------------------------- | -------------------------------------- |
| [Obsidian Help Docs](https://help.obsidian.md)                | Official syntax, plugin API            |
| [Obsidian Forum](https://forum.obsidian.md)                   | Community workflows thực tế            |
| Tiago Forte — _Building a Second Brain_                       | PARA method, Progressive Summarization |
| Sönke Ahrens — _How to Take Smart Notes_                      | Zettelkasten methodology               |
| [LYT Kit](https://www.linkingyourthinking.com/) của Nick Milo | MOC, Evergreen notes                   |
| [Obsidian Roundup](https://obsidianroundup.org/)              | Newsletter tổng hợp tips               |

### Về Learning Science (để AI dạy đúng cách)

| Nguồn                                                                                   | Nội dung                                            |
| --------------------------------------------------------------------------------------- | --------------------------------------------------- |
| _Make It Stick_ — Brown et al.                                                          | Spaced repetition, interleaving, retrieval practice |
| [Coursera: Learning How to Learn](https://www.coursera.org/learn/learning-how-to-learn) | Pomodoro, chunking, diffuse thinking                |
| Bloom's Taxonomy                                                                        | Framework phân cấp độ hiểu biết                     |
| Andy Matuschak's [Working Notes](https://notes.andymatuschak.org)                       | Evergreen notes philosophy                          |

### Về Cursor/Claude Skill Engineering

| Nguồn                                                                   | Nội dung                         |
| ----------------------------------------------------------------------- | -------------------------------- |
| Anthropic Prompt Engineering docs                                       | Cách viết system prompt hiệu quả |
| `.cursorrules` community repos trên GitHub                              | Ví dụ rules thực tế              |
| [Awesome CursorRules](https://github.com/PatrickJS/awesome-cursorrules) | Tập hợp rules chất lượng         |
| Các SKILL.md mẫu trong `/mnt/skills/`                                   | Pattern hiện tại đang dùng       |

### Về Obsidian Plugin API (nếu muốn tự động hóa)

- [Obsidian Plugin Developer Docs](https://docs.obsidian.md/Plugins/Getting+started/Build+a+plugin)
- Dataview plugin docs: query syntax để Claude generate đúng
- Templater docs: để Claude tạo template JS hợp lệ

---

## 🏗️ Gợi ý Cấu Trúc File Skill

```
/skills/obsidian-learning/
├── SKILL.md                    ← Router chính, mô tả khi nào dùng gì
├── note-templates/
│   ├── concept-note.md
│   ├── literature-note.md
│   └── moc-template.md
├── conventions.md              ← Frontmatter schema, folder structure
├── plugin-syntax.md            ← Dataview, Templater, Tasks syntax
└── pedagogy-guide.md           ← Bloom's taxonomy, retrieval practice rules
```

---
