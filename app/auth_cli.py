"""
Utility CLI for managing the local credential store.

Usage examples:
    python -m auth_cli list
    python -m auth_cli add --username alice --role admin
    python -m auth_cli password --username alice
    python -m auth_cli delete --username alice
"""

from __future__ import annotations

import argparse
import getpass
import sys
from typing import Iterable

from auth import hash_password
from auth_store import UserRecord, list_users, remove_user, upsert_user
from config import AUTH_ADMIN_OS_USERS


def _require_admin_os_user() -> None:
    current_user = getpass.getuser()
    if current_user not in AUTH_ADMIN_OS_USERS:
        allowed = ", ".join(AUTH_ADMIN_OS_USERS) or "<未配置>"
        print(
            f"当前系统用户“{current_user}”无权修改账号。\n"
            f"允许的系统用户: {allowed}\n"
            "如需授权，请在运行前设置环境变量 AUTH_ADMIN_OS_USERS 或更新 config.py。",
            file=sys.stderr,
        )
        sys.exit(1)


def _prompt_password(confirm: bool = True) -> str:
    while True:
        pwd = getpass.getpass("请输入密码: ")
        if not pwd:
            print("密码不能为空。", file=sys.stderr)
            continue
        if not confirm:
            return pwd
        pwd_confirm = getpass.getpass("请再次输入密码: ")
        if pwd == pwd_confirm:
            return pwd
        print("两次输入的密码不一致，请重试。", file=sys.stderr)


def cmd_list(_: argparse.Namespace) -> None:
    rows: Iterable[UserRecord] = list_users()
    if not rows:
        print("当前没有用户。")
        return
    for record in rows:
        print(f"{record.username}\t{record.role}")


def cmd_add(args: argparse.Namespace) -> None:
    _require_admin_os_user()
    password = args.password or _prompt_password()
    hashed = hash_password(password)
    upsert_user(UserRecord(username=args.username, password_hash=hashed, role=args.role))
    print(f"用户 {args.username} 已添加/更新。")


def cmd_password(args: argparse.Namespace) -> None:
    _require_admin_os_user()
    password = args.password or _prompt_password()
    hashed = hash_password(password)
    upsert_user(UserRecord(username=args.username, password_hash=hashed, role=args.role))
    print(f"用户 {args.username} 密码已更新。")


def cmd_delete(args: argparse.Namespace) -> None:
    _require_admin_os_user()
    if remove_user(args.username):
        print(f"用户 {args.username} 已删除。")
    else:
        print(f"未找到用户 {args.username}。", file=sys.stderr)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地账号管理工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="列出所有用户")
    list_parser.set_defaults(func=cmd_list)

    add_parser = subparsers.add_parser("add", help="新增或更新用户")
    add_parser.add_argument("--username", required=True, help="用户名")
    add_parser.add_argument("--role", default="user", help="角色")
    add_parser.add_argument("--password", help="明文密码（不推荐）")
    add_parser.set_defaults(func=cmd_add)

    password_parser = subparsers.add_parser("password", help="更新用户密码")
    password_parser.add_argument("--username", required=True, help="用户名")
    password_parser.add_argument("--role", default="user", help="角色")
    password_parser.add_argument("--password", help="明文密码（不推荐）")
    password_parser.set_defaults(func=cmd_password)

    delete_parser = subparsers.add_parser("delete", help="删除用户")
    delete_parser.add_argument("--username", required=True, help="用户名")
    delete_parser.set_defaults(func=cmd_delete)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

