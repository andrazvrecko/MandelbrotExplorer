#pragma once
#include "Screen.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <thread>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#define WIDTH 1280
#define HEIGHT 720

class MandelbrotExplorer : public olc::PixelGameEngine
{
public:
	MandelbrotExplorer() {
		sAppName = "Mandelbrot Explorer";
	}
	bool OnUserCreate() override;
	bool OnUserDestroy() override;
	bool OnUserUpdate(float fElapsedTime) override;

	void simpleFractal(const olc::vi2d& t_pixTopLeft, const olc::vi2d& t_pixBotRight, const olc::vd2d& t_fracTopLeft, const olc::vd2d& t_fracBotRight, const int& iterations);
	void fractalWithThreads(const olc::vi2d& t_pixTopLeft, const olc::vi2d& t_pixBotRight, const olc::vd2d& t_fracTopLeft, const olc::vd2d& t_fracBotRight, const int& iterations);
	void cudaSimpleFractal(int tlX, int tlY, int brX, int brY, double f_tlX, double f_tlY, double f_brX, double f_brY, int iterations);
	void cudaSimpleJulia(int tlX, int tlY, int brX, int brY, double f_tlX, double f_tlY, double f_brX, double f_brY, int iterations);


	void WorldToScreen(const olc::vd2d& v, olc::vi2d& n)
	{
		n.x = (int)((v.x - vOffset.x) * vScale.x);
		n.y = (int)((v.y - vOffset.y) * vScale.y);
	}

	void ScreenToWorld(const olc::vi2d& n, olc::vd2d& v)
	{
		v.x = (double)(n.x) / vScale.x + vOffset.x;
		v.y = (double)(n.y) / vScale.y + vOffset.y;
	}

private:
	olc::vd2d vOffset = { 0.0, 0.0 };
	olc::vd2d vStartPan = { 0.0, 0.0 };
	olc::vd2d vScale = { WIDTH / 2.0, HEIGHT };
	int* m_pFractal;
	std::vector<std::vector<int>>* m_pVector;
	int m_Mode = 1;
	int m_Set = 0;
	int m_drawMode = 1;
	int m_Iterations = 768;
	bool m_bDebug = false;
};

