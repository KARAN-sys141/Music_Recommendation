import pathlib, re

path = pathlib.Path('templates/song_detail.html')
content = path.read_text(encoding='utf-8')

content = content.replace('id="mainPlayBtn"', 'id="songArtPlayBtn"')

content = content.replace('getElementById("mainPlayBtn")', 'getElementById("songArtPlayBtn")')

content = content.replace(
    'const mainPlayBtn = document.getElementById',
    'const songArtPlayBtn = document.getElementById'
)

content = content.replace('if (mainPlayBtn)', 'if (songArtPlayBtn)')
content = content.replace(', mainPlayBtn)', ', songArtPlayBtn)')
content = content.replace('mainPlayBtn.addEventListener', 'songArtPlayBtn.addEventListener')

path.write_text(content, encoding='utf-8')

remaining = [m.start() for m in re.finditer('mainPlayBtn', content)]
print('Remaining mainPlayBtn refs:', len(remaining))
for pos in remaining:
    print(' ', repr(content[max(0,pos-20):pos+60]))
