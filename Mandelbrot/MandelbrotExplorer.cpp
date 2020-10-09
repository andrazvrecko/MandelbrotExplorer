#include "MandelbrotExplorer.h"

bool MandelbrotExplorer::OnUserCreate()
{
	m_pFractal = new int[ScreenWidth() * ScreenHeight()]{ 0 };
	
	m_pVector = new std::vector<std::vector<int> >(ScreenWidth(), std::vector<int>(ScreenHeight()));
	
	for (int i = 0; i < ScreenWidth(); i++) {
		for (int j = 0; j < ScreenHeight(); j++) {
			m_pVector->at(i).at(j) = 0;
		}
	}
	return true;
}

bool MandelbrotExplorer::OnUserDestroy()
{
	return true;
}

bool MandelbrotExplorer::OnUserUpdate(float fElapsedTime)
{
	olc::vd2d vMouse = { (double)GetMouseX(), (double)GetMouseY() };

	if (GetMouse(0).bPressed)
	{
		vStartPan = vMouse;
	}

	if (GetMouse(0).bHeld)
	{
		vOffset -= (vMouse - vStartPan) / vScale;
		vStartPan = vMouse;
	}

	olc::vd2d vMouseBeforeZoom;
	ScreenToWorld(vMouse, vMouseBeforeZoom);

	if (GetKey(olc::Key::Q).bHeld || GetMouseWheel() > 0) vScale *= 1.1;
	if (GetKey(olc::Key::A).bHeld || GetMouseWheel() < 0) vScale *= 0.9;
	if (GetKey(olc::Key::K).bPressed) m_bDebug = !m_bDebug;
	if (GetKey(olc::Key::K0).bPressed) m_Mode = 0;
	if (GetKey(olc::Key::K1).bPressed) m_Mode = 1;
	if (GetKey(olc::UP).bPressed) m_Iterations += 64;
	if (GetKey(olc::DOWN).bPressed) m_Iterations -= 64;
	if (m_Iterations < 64) m_Iterations = 64;

	olc::vd2d vMouseAfterZoom;
	ScreenToWorld(vMouse, vMouseAfterZoom);
	vOffset += (vMouseBeforeZoom - vMouseAfterZoom);

	olc::vi2d c_pixTopLeft = { 0,0 };
	olc::vi2d c_pixBotRight = { ScreenWidth(), ScreenHeight() };
	olc::vd2d c_fracTopLeft = { -2.0, -1.0 };
	olc::vd2d c_fracBotRight = { 1.0, 1.0 };

	ScreenToWorld(c_pixTopLeft, c_fracTopLeft);
	ScreenToWorld(c_pixBotRight, c_fracBotRight);

	//Start timer
	auto tp1 = std::chrono::high_resolution_clock::now();
	
	switch (m_Mode) {
	case 0: 
		simpleFractal(c_pixTopLeft, c_pixBotRight, c_fracTopLeft, c_fracBotRight, m_Iterations);
		break;
	case 1:
		fractalWithThreads(c_pixTopLeft, c_pixBotRight, c_fracTopLeft, c_fracBotRight, m_Iterations);
		break;
	}
	//simpleFractal(c_pixTopLeft, c_pixBotRight, c_fracTopLeft, c_fracBotRight, m_Iterations);
	//fractalWithThreads(c_pixTopLeft, c_pixBotRight, c_fracTopLeft, c_fracBotRight, m_Iterations);


	for (int x = 0; x < ScreenWidth(); x++)
	{
		for (int y = 0; y < ScreenHeight(); y++)
		{
			float a = 0.1f;
			float n = m_pVector->at(x).at(y);
			//Draw(x, y, olc::PixelF(122, 122, 122));
			Draw(x, y, olc::PixelF(0.5f * sin(a * n) + 0.5f, 0.5f * sin(a * n + 2.094f) + 0.5f, 0.5f * sin(a * n + 4.188f) + 0.5f));
		}
	}

	//Stop timer
	auto tp2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedTime = tp2 - tp1;

	if (m_bDebug) {
		DrawString(0, 30, "Time Taken: " + std::to_string(elapsedTime.count()) + "s", olc::WHITE, 3);
		DrawString(0, 60, "Mode: " + std::to_string(m_Mode));
		DrawString(0, 70, "Iterations: " + std::to_string(m_Iterations));
		DrawString(0, 80, "Top Left Fractal: " + std::to_string(c_fracTopLeft.x) + " - " + std::to_string(c_fracTopLeft.y));
		DrawString(0, 90, "Bot Right Fractal: " + std::to_string(c_fracBotRight.x) + " - " + std::to_string(c_fracBotRight.y));
		DrawString(0, 100, "Top Left Fractal: " + std::to_string(c_pixTopLeft.x) + " - " + std::to_string(c_pixTopLeft.y));
		DrawString(0, 110, "Bot Right Fractal: " + std::to_string(c_pixBotRight.x) + " - " + std::to_string(c_pixBotRight.y));
	}
	return true;
}

void MandelbrotExplorer::simpleFractal(const olc::vi2d& t_pixTopLeft, const olc::vi2d& t_pixBotRight, const olc::vd2d& t_fracTopLeft, const olc::vd2d& t_fracBotRight, const int& iterations)
{
	double x0 = (t_fracBotRight.x - t_fracTopLeft.x) / (double(t_pixBotRight.x) - double(t_pixTopLeft.x));
	double y0 = (t_fracBotRight.y - t_fracTopLeft.y) / (double(t_pixBotRight.y) - double(t_pixTopLeft.y));
	double coordX = t_fracTopLeft.x, coordY = t_fracTopLeft.y;
	double x = 0, y = 0, cIteration = 0, xtemp;
	
	for (int i = t_pixTopLeft.x; i < t_pixBotRight.x; i++) {
		coordY = t_fracTopLeft.y;
		for (int j = t_pixTopLeft.y; j < t_pixBotRight.y; j++) {
			x = 0;
			y = 0;
			cIteration = 0;
			while (x*x + y*y <= 4 && cIteration < iterations)
			{
				xtemp = x * x - y * y + coordX;
				y = 2 * x * y + coordY;
				x = xtemp;
				cIteration += 1;
			}
			m_pVector->at(i).at(j) = cIteration;
			coordY += y0;
		}
		coordX += x0;
	}

}

void MandelbrotExplorer::fractalWithThreads(const olc::vi2d& t_pixTopLeft, const olc::vi2d& t_pixBotRight, const olc::vd2d& t_fracTopLeft, const olc::vd2d& t_fracBotRight, const int& iterations)
{
	const int maxThreads = 32;
	std::thread threads[maxThreads];
	int widthFactor = (t_pixBotRight.x - t_pixTopLeft.x) / maxThreads;
	double fracWidthFactor = (t_fracBotRight.x - t_fracTopLeft.x) / double(maxThreads);
	
	for (int i = 0; i < maxThreads; i++) {
		threads[i] = std::thread(&MandelbrotExplorer::simpleFractal, this,
			olc::vi2d(t_pixTopLeft.x + widthFactor * (i), t_pixTopLeft.y),
			olc::vi2d(t_pixTopLeft.x + widthFactor * (i + 1), t_pixBotRight.y),
			olc::vd2d(t_fracTopLeft.x + fracWidthFactor * double(i), t_fracTopLeft.y),
			olc::vd2d(t_fracTopLeft.x + fracWidthFactor * double(i + 1), t_fracBotRight.y),
			iterations);
	}

	for (int i = 0; i < maxThreads; i++) {
		threads[i].join();
	}

}
