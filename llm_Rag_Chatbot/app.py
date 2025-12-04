from core.sqlite_patch import patch_sqlite
patch_sqlite()

from core.ui import launch_ui

if __name__ == "__main__":
    # 시작 시 URL 자동 색인 제거 — 업로드 후 색인하도록 UI만 띄움
    launch_ui()
