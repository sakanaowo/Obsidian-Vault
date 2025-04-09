const fs = require("fs");
const path = require("path");

const outputDir = "cloned-structure";

// Xử lý nếu tồn tại: xoá file hoặc dọn thư mục cũ
if (fs.existsSync(outputDir)) {
    const stat = fs.statSync(outputDir);
    if (stat.isFile()) {
        fs.unlinkSync(outputDir); // Xoá nếu là file
    } else {
        fs.rmSync(outputDir, { recursive: true, force: true }); // Xoá toàn bộ thư mục cũ
    }
}

// Tạo lại thư mục gốc
fs.mkdirSync(outputDir);

const treeInput = `
├── backend
│   ├── package.json
│   ├── package-lock.json
│   └── src
│       ├── controllers
│       │   ├── auth.controller.js
│       │   └── message.controller.js
│       ├── index.js
│       ├── lib
│       │   ├── cloudinary.js
│       │   ├── db.js
│       │   ├── socket.js
│       │   └── utils.js
│       ├── middleware
│       │   └── auth.middleware.js
│       ├── models
│       │   ├── message.model.js
│       │   └── user.model.js
│       └── routes
│           ├── auth.route.js
│           └── message.route.js
└── frontend
    ├── eslint.config.js
    ├── index.html
    ├── package.json
    ├── package-lock.json
    ├── public
    │   ├── avatar.png
    │   └── vite.svg
    ├── src
    │   ├── App.jsx
    │   ├── components
    │   │   ├── AuthImagePattern.jsx
    │   │   ├── ChatContainer.jsx
    │   │   ├── ChatHeader.jsx
    │   │   ├── MessageInput.jsx
    │   │   ├── Navbar.jsx
    │   │   ├── NoChatSelected.jsx
    │   │   ├── Sidebar.jsx
    │   │   └── skeletons
    │   │       ├── MessageSkeleton.jsx
    │   │       └── SidebarSkeleton.jsx
    │   ├── constants
    │   │   └── index.js
    │   ├── index.css
    │   ├── lib
    │   │   ├── axios.js
    │   │   └── utils.js
    │   ├── main.jsx
    │   ├── pages
    │   │   ├── HomePage.jsx
    │   │   ├── LoginPage.jsx
    │   │   ├── ProfilePage.jsx
    │   │   ├── SettingsPage.jsx
    │   │   └── SignUpPage.jsx
    │   └── store
    │       ├── useAuthStore.js
    │       ├── useChatStore.js
    │       └── useThemeStore.js
    └── vite.config.js
`;

function createFromTree(treeString, outputDir = "cloned-structure") {
    const lines = treeString.trim().split("\n");
    let stack = [];
    let lastDepth = 0;

    lines.forEach((line) => {
        const depth = (line.match(/^\s*/)[0].length) / 4;
        const name = line.replace(/^.*[└├]── /, "").trim();
        const isDir = !name.includes(".");

        if (depth <= lastDepth) {
            stack = stack.slice(0, depth);
        }
        lastDepth = depth;

        const currentPath = path.join(outputDir, ...stack, name);
        if (isDir) {
            fs.mkdirSync(currentPath, { recursive: true });
            stack.push(name);
        } else {
            const fileName = name.replace(/\.[^/.]+$/, ".md"); // Đổi đuôi thành .md
            const filePath = path.join(outputDir, ...stack, fileName);
            fs.writeFileSync(filePath, `# ${fileName}\n`);
        }
    });

    console.log(`✅ Project recreated in: ${outputDir}`);
}

createFromTree(treeInput);
