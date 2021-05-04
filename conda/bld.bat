if "%INSTALL_PREFIX%" == "" set INSTALL_PREFIX=%cd%\scipp_install & call tools\make_and_install.bat

move %INSTALL_PREFIX%\scippneutron %CONDA_PREFIX%\lib\
move %INSTALL_PREFIX%\bin\scippneutron*.dll %CONDA_PREFIX%\bin\
move %INSTALL_PREFIX%\Lib\scippneutron*.lib %CONDA_PREFIX%\Lib\
move %INSTALL_PREFIX%\Lib\cmake\scippneutron %CONDA_PREFIX%\Lib\cmake\
move %INSTALL_PREFIX%\include\scippneutron* %CONDA_PREFIX%\include\
