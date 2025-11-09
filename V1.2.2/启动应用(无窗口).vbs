Set WshShell = CreateObject("WScript.Shell")
' 获取脚本所在目录
Set fso = CreateObject("Scripting.FileSystemObject")
scriptPath = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = scriptPath & "\run.bat"

' 隐藏窗口运行bat文件 (0 = 隐藏窗口)
WshShell.Run """" & batPath & """", 0, False

Set fso = Nothing
Set WshShell = Nothing
