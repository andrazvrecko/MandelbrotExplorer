#define OLC_PGE_APPLICATION
#include "MandelbrotExplorerCuda.cuh"

int main()
{
	MandelbrotExplorer demo;
	if (demo.Construct(WIDTH, HEIGHT, 1, 1, false, false))
		demo.Start();
	return 0;
}