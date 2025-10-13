mkdir buildRelease
cd buildRelease
cmake -DTORCH_CONFIGURATION=Release  -DOneConfigOnly=ON ..
pause