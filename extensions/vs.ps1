# Путь к папке с Visual Studio
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community"

# Скрипт для настройки окружения C++
$vcvarsPath = "$vsPath\VC\Auxiliary\Build\vcvarsall.bat"
if (Test-Path $vcvarsPath) {
    # Импортируем переменные окружения из bat-файла в текущую сессию PowerShell
    cmd /c "`"$vcvarsPath`" x64 && set" | ForEach-Object {
        if ($_ -match "^(.*?)=(.*)$") {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2] -ErrorAction SilentlyContinue
        }
    }
} else {
    Write-Error "vcvarsall.bat not found. Check the Visual Studio path."
}