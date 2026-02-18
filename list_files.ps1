Get-ChildItem 'F:\BUREAU\disk_cleaner' -Recurse -File -Exclude '*.pyc' |
    Where-Object { $_.DirectoryName -notlike '*__pycache__*' -and $_.DirectoryName -notlike '*.venv*' } |
    Select-Object FullName, Length |
    Format-Table -AutoSize
