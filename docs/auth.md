# 🔐 本地登录认证配置指南

应用默认在 `app/config/users.json` 中查找用户凭据，每个账号仅保存 bcrypt 哈希（不会保存明文密码）。下列步骤可帮助你初始化账号、维护密码并了解安全注意事项。

## 1. 初始化 & 查看用户

```bash
cd Sth-in-EB
python -m app.auth_cli list
```

- 若文件不存在，系统会自动创建结构：  
  ```json
  {
    "users": []
  }
  ```
- `list` 命令会显示所有用户名和角色。

## 2. 创建或更新账号

```bash
python -m app.auth_cli add --username admin --role admin
```

命令会提示输入两次密码并自动写入哈希。若账号已存在，将更新其密码和角色。  
也可通过 `--password` 直接传入明文（不推荐，仅脚本化时使用）。

## 3. 重设密码

```bash
python -m app.auth_cli password --username alice
```

该命令与 `add` 类似，但语义上用于重置密码。

## 4. 删除账号

```bash
python -m app.auth_cli delete --username guest
```

返回 0 表示删除成功；若账号不存在则会返回非 0。

## 5. 运行应用

启动 `streamlit run app/app.py` 或 `run.bat` 后：

1. 首个访问者看到登录界面，需输入上一步创建的用户名/密码。
2. 登录成功后，`st.session_state` 将保存 `auth.is_authenticated` 与 `auth.username` 用于整个应用和所有页面。
3. 侧边栏包含“退出登录”按钮，可立即清除会话并返回登录界面。

## 6. 安全建议

- `users.json` 仅适合少量内部用户，请结合文件系统权限限制访问。
- 定期轮换密码：重复执行 `python -m app.auth_cli password ...`。
- 需要多人协作时，可使用版本控制忽略 `app/config/users.json`，避免在仓库中提交真实账号。
- 如果需要更强的安全策略（如多因素或集中式身份认证），可在今后替换 `auth.py` 中的 `authenticate` 实现。

## 7. 限制谁能修改账号

- 程序会读取 `config.AUTH_ADMIN_OS_USERS`（默认值在 `app/config/config.py`）或环境变量 `AUTH_ADMIN_OS_USERS`（逗号分隔多个用户名）。  
- 当系统用户名不在允许列表里时，`add/password/delete` 命令会拒绝执行并提示“无权修改账号”。  
- 若需要在其他电脑上维护账号，可在启动终端前设置：
  ```powershell
  set AUTH_ADMIN_OS_USERS=24561,alice
  ```
  或直接修改 `config.py` 中的 `DEFAULT_AUTH_ADMIN_USERS`。

