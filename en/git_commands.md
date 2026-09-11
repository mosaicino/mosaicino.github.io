从零开始，最常见的这条路：

**一次性设置**（每台机器只做一次）

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

**第一次把项目传上去**

```bash
cd 你的项目目录
git init
git add .
git commit -m "初始版本"
```

然后去 GitHub 网页上建一个空仓库（**不要**勾选 Add README，否则会冲突），建完页面会给你一个地址，回到命令行：

```bash
git remote add origin https://github.com/你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

**以后每次改完**，就只有三行：

```bash
git add .
git commit -m "说明这次改了什么"
git push
```

几个容易卡住的点：

- **推送时要密码**——GitHub 早就不收账号密码了，得用 Personal Access Token。在 GitHub 的 Settings → Developer settings → Personal access tokens 里生成一个，勾 `repo` 权限，然后把它当密码粘进去。生成后只显示一次，记得存下来。
- **`git push` 报 rejected**——远程有你本地没有的提交，先 `git pull --rebase` 再 push。
- **不小心把不该传的传上去了**——先建 `.gitignore` 再 `git add`。你这个静态网站至少该忽略 `.DS_Store` 和 `node_modules/`。

顺带一提，既然是 HTML5 UP 的静态模板，可以直接用 **GitHub Pages** 白嫖托管：仓库 Settings → Pages → Source 选 `main` 分支根目录，等一两分钟就能通过 `你的用户名.github.io/仓库名` 访问。之后每次 push 自动更新，不用管服务器。

如果你用的是 VS Code，左侧那个分支图标的面板可以点着完成上面全部操作，不用记命令。