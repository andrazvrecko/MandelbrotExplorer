#include "MandelbrotExplorer.h"

bool MandelbrotExplorer::OnUserCreate()
{
	m_pFractal = new int[ScreenWidth() * ScreenHeight()]{ 0 };
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

	olc::vd2d vMouseAfterZoom;
	ScreenToWorld(vMouse, vMouseAfterZoom);
	vOffset += (vMouseBeforeZoom - vMouseAfterZoom);

	olc::vi2d c_pixTopLeft = { 0,0 };
	olc::vi2d c_pixBotRight = { ScreenWidth(), ScreenHeight() };
	olc::vd2d c_fracTopLeft = { 0.0, 0.0 };
	olc::vd2d c_fracBotRight = { 0.0, 0.0 };

	ScreenToWorld(c_pixTopLeft, c_fracTopLeft);
	ScreenToWorld(c_pixBotRight, c_fracBotRight);

	//Start timer
	auto tp1 = std::chrono::high_resolution_clock::now();
	
	simpleFractal(c_pixTopLeft, c_pixBotRight, c_fracTopLeft, c_fracBotRight, m_Iterations);
	for (int y = 0; y < ScreenHeight(); y++)
	{
		for (int x = 0; x < ScreenWidth(); x++)
		{
			Draw(x, y, olc::PixelF(100,100,100));
		}
	}

	//Stop timer
	auto tp2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedTime = tp2 - tp1;

	if (m_bDebug) {
		DrawString(0, 30, "Time Taken: " + std::to_string(elapsedTime.count()) + "s", olc::WHITE, 3);
		DrawString(0, 80, "Top Left Fractal: " + std::to_string(c_fracTopLeft.x) + " - " + std::to_string(c_fracTopLeft.y));
		DrawString(0, 90, "Bot Right Fractal: " + std::to_string(c_fracBotRight.x) + " - " + std::to_string(c_fracBotRight.y));
		DrawString(0, 100, "Top Left Fractal: " + std::to_string(c_pixTopLeft.x) + " - " + std::to_string(c_pixTopLeft.y));
		DrawString(0, 110, "Bot Right Fractal: " + std::to_string(c_pixBotRight.x) + " - " + std::to_string(c_pixBotRight.y));
	}
	return true;
}

void MandelbrotExplorer::simpleFractal(const olc::vi2d& t_pixTopLeft, const olc::vi2d& t_pixBotRight, const olc::vd2d& t_fracTopLeft, const olc::vd2d& t_fracBotRight, const int& iterations)
{

}
