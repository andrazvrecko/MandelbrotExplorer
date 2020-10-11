#include "MandelbrotExplorerCuda.cuh"

__global__
void cudaFractalLoop(int *buffer, int iterations, double scaleX, double scaleY, double offsetX, double offsetY) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int it = 0;
	
	double coordX = (double)(col) / scaleX + offsetX;
	double coordY = (double)(row) / scaleY + offsetY;
		
	double x = 0.0f;
	double y = 0.0f;
	double xtemp;
	while (x * x + y * y <= 4 && it < iterations)
	{
		xtemp = x * x - y * y + coordX;
		y = 2 * x * y + coordY;
		x = xtemp;
		it++;
	}
	int index = row * WIDTH + col;
	buffer[index] = it;
}

__global__
void cudaJuliaLoop(int* buffer, int iterations, double scaleX, double scaleY, double offsetX, double offsetY) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int it = 0;

	double coordX = (double)(col) / scaleX + offsetX;
	double coordY = (double)(row) / scaleY + offsetY;

	double x = 0.0f;
	double y = 0.0f;
	double xtemp;
	while (coordX * coordX + coordY + coordY < 4 && it < iterations) {
		xtemp = coordX * coordX - coordY * coordY;
		coordY = 2 * coordX * coordY + 0.58;
		coordX = xtemp + 0.282;
		it++;
	}
	int index = row * WIDTH + col;
	buffer[index] = it;
}

void MandelbrotExplorer::cudaSimpleFractal(int tlX, int tlY, int brX, int brY, double f_tlX, double f_tlY, double f_brX, double f_brY, int iterations)
{
	dim3 block_size(16, 16);
	dim3 grid_size(WIDTH / block_size.x, HEIGHT / block_size.y);


	cudaFractalLoop<<<grid_size, block_size>>>(m_pFractal, iterations, vScale.x, vScale.y, vOffset.x, vOffset.y);
	cudaDeviceSynchronize();

}

void MandelbrotExplorer::cudaSimpleJulia(int tlX, int tlY, int brX, int brY, double f_tlX, double f_tlY, double f_brX, double f_brY, int iterations)
{
	dim3 block_size(16, 16);
	dim3 grid_size(WIDTH / block_size.x, HEIGHT / block_size.y);


	cudaJuliaLoop << <grid_size, block_size >> > (m_pFractal, iterations, vScale.x, vScale.y, vOffset.x, vOffset.y);
	cudaDeviceSynchronize();

}

bool MandelbrotExplorer::OnUserCreate()
{
	cudaMallocManaged(&m_pFractal, size_t(ScreenWidth()) * size_t(ScreenHeight()) * sizeof(int));
	
	m_pVector = new std::vector<std::vector<int>> (ScreenWidth(), std::vector<int>(ScreenHeight()));

	for (int i = 0; i < ScreenWidth(); i++) {
		for (int j = 0; j < ScreenHeight(); j++) {
			m_pVector->at(i).at(j) = 0;
		}
	}
	return true;
}

bool MandelbrotExplorer::OnUserDestroy()
{
	cudaFree(m_pFractal);
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
	if (GetKey(olc::Key::J).bPressed) m_Set = 1;
	if (GetKey(olc::Key::M).bPressed) m_Set = 0;
	if (GetKey(olc::Key::K0).bPressed) m_Mode = 0;
	if (GetKey(olc::Key::K1).bPressed) m_Mode = 1;
	if (GetKey(olc::Key::K2).bPressed) m_Mode = 2;
	if (GetKey(olc::Key::K7).bPressed) m_drawMode = 0;
	if (GetKey(olc::Key::K8).bPressed) m_drawMode = 1;
	if (GetKey(olc::Key::K9).bPressed) m_drawMode = 2;
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

	switch (m_Set) {
	case 0:
		switch (m_Mode) {
		case 0:
			simpleFractal(c_pixTopLeft, c_pixBotRight, c_fracTopLeft, c_fracBotRight, m_Iterations);
			break;
		case 1:
			fractalWithThreads(c_pixTopLeft, c_pixBotRight, c_fracTopLeft, c_fracBotRight, m_Iterations);
			break;
		case 2:

			int tlX, tlY, brX, brY;
			double f_tlX, f_tlY, f_brX, f_brY;
			tlX = c_pixTopLeft.x;
			tlY = c_pixTopLeft.y;
			brX = c_pixBotRight.x;
			brY = c_pixBotRight.y;
			f_tlX = c_fracTopLeft.x;
			f_tlY = c_fracTopLeft.y;
			f_brX = c_fracBotRight.x;
			f_brY = c_fracBotRight.y;

			cudaSimpleFractal(tlX, tlY, brX, brY, f_tlX, f_tlY, f_brX, f_brY, m_Iterations);
			break;
		}
		break;
	case 1:
		int tlX, tlY, brX, brY;
		double f_tlX, f_tlY, f_brX, f_brY;
		tlX = c_pixTopLeft.x;
		tlY = c_pixTopLeft.y;
		brX = c_pixBotRight.x;
		brY = c_pixBotRight.y;
		f_tlX = c_fracTopLeft.x;
		f_tlY = c_fracTopLeft.y;
		f_brX = c_fracBotRight.x;
		f_brY = c_fracBotRight.y;

		cudaSimpleJulia(tlX, tlY, brX, brY, f_tlX, f_tlY, f_brX, f_brY, m_Iterations);
		break;
	}
	for (int x = 0; x < ScreenWidth(); x++)
	{
		for (int y = 0; y < ScreenHeight(); y++)
		{
			float a = 0.1f;
			float n;
			if (m_Mode != 2) {
				n = m_pVector->at(x).at(y);
			}
			else {
				n = (float)m_pFractal[y * ScreenWidth() + x];
			}
			
			switch (m_drawMode) {
			case 0:
				Draw(x, y, olc::PixelF(0.5f * sin(a * n) + 0.5f, 0.5f * sin(a * n + 2.094f) + 0.5f, 0.5f * sin(a * n + 4.188f) + 0.5f));
				break;
			case 1:
				Draw(x, y, olc::PixelF(255 - n, 255 - n, 255 - n));
				break;
			case 2:
				break;
				Draw(x, y, olc::PixelF(n, n, n));
			}
		}
	}

	auto tp2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedTime = tp2 - tp1;

	if (m_bDebug) {
		DrawString(0, 20, "Time Taken: " + std::to_string(elapsedTime.count()) + "s", olc::WHITE, 2);
		DrawString(0, 70, "Iterations: " + std::to_string(m_Iterations));
		DrawString(0, 80, "Top Left Fractal: " + std::to_string(c_fracTopLeft.x) + " - " + std::to_string(c_fracTopLeft.y));
		DrawString(0, 90, "Bot Right Fractal: " + std::to_string(c_fracBotRight.x) + " - " + std::to_string(c_fracBotRight.y));
		DrawString(0, 100, "Top Left Fractal: " + std::to_string(c_pixTopLeft.x) + " - " + std::to_string(c_pixTopLeft.y));
		DrawString(0, 110, "Bot Right Fractal: " + std::to_string(c_pixBotRight.x) + " - " + std::to_string(c_pixBotRight.y));
		
		if (m_Set == 0) {
			DrawString(0, 50, "Mandelbrot Set ");
			if (m_Mode == 0) {
				DrawString(0, 60, "Mode: Simple");
			}
			else if (m_Mode == 1) {
				DrawString(0, 60, "Mode: Threads");
			}
			else {
				DrawString(0, 60, "Mode: Cuda");
			}
		}
		else {
			DrawString(0, 60, "Julia Set ");
		}
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
			while (x * x + y * y <= 4 && cIteration < iterations)
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



