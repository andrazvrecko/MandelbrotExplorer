#include "MandelbrotExplorer.h"

bool MandelbrotExplorer::OnUserCreate()
{
	return true;
}

bool MandelbrotExplorer::OnUserDestroy()
{
	return true;
}

bool MandelbrotExplorer::OnUserUpdate(float fElapsedTime)
{
	for (int y = 0; y < ScreenHeight(); y++)
	{
		for (int x = 0; x < ScreenWidth(); x++)
		{
			Draw(x, y, olc::Pixel(122, 122, 122));
		}
	}
	return true;
}
