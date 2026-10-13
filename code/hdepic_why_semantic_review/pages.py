"""Generate four-case inspection pages from existing sixteen-frame sheets."""
from pathlib import Path
from PIL import Image
from .prepare import OUT, rows

def pages():
    for batch in range(1,4):
        cases=rows(OUT/'blinded'/f'batch{batch}.jsonl')
        folder=OUT/'blinded'/f'inspection_batch{batch}';folder.mkdir(exist_ok=True)
        for start in range(0,len(cases),4):
            group=cases[start:start+4]
            dest=folder/f'{group[0]["case"]}-{group[-1]["case"]}.jpg'
            if dest.exists():continue
            sheet=Image.new('RGB',(1604,1818),'white')
            for i,c in enumerate(group):
                with Image.open(OUT/'blinded'/c['contact_sheet']) as image:
                    image=image.resize((802,909),Image.Resampling.LANCZOS)
                    sheet.paste(image,((i%2)*802,(i//2)*909))
            sheet.save(dest,quality=94,subsampling=0)

if __name__=='__main__':pages()
