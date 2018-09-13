set count=0

:Repeat1

python phase1_2.py "poiMAP_Dunkin' Donuts_500m_16part"

set /a count = count + 1

if not "%count%" == "10" goto Repeat1


