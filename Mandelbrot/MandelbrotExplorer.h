#pragma once
#include "Screen.h"

class MandelbrotExplorer : public olc::PixelGameEngine
{
public:
	MandelbrotExplorer() {
		sAppName = "Mandelbrot Explorer";
	}
	bool OnUserCreate() override;
	bool OnUserDestroy() override;
	bool OnUserUpdate(float fElapsedTime) override;

private:
};

