#define OLC_PGE_APPLICATION
#include "MandelbrotExplorer.h"

int main()
{
	MandelbrotExplorer demo;
	if (demo.Construct(1280, 720, 1, 1, false, false))
		demo.Start();
	return 0;
} 