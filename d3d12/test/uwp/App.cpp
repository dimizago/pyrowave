// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Xbox (UWP) front-end for the D3D12 decoder test: replays the test vectors packaged
// under vectors\ (dumped on a PC with pyrowave-d3d12-decode-compare --dump), shows
// the report in a dialog and writes it to LocalState\pyrowave_d3d12_test.txt, which
// Device Portal's File explorer can download.

#include "vector_test.hpp"

#include <fstream>
#include <stdlib.h>
#include <thread>

#include <winrt/Windows.ApplicationModel.Activation.h>
#include <winrt/Windows.ApplicationModel.Core.h>
#include <winrt/Windows.ApplicationModel.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.UI.Core.h>
#include <winrt/Windows.UI.Popups.h>

using namespace winrt;
using namespace Windows::ApplicationModel;
using namespace Windows::ApplicationModel::Activation;
using namespace Windows::ApplicationModel::Core;
using namespace Windows::Storage;
using namespace Windows::UI::Core;
using namespace Windows::UI::Popups;

namespace
{
const wchar_t *const vector_files[] = {
	L"1920x1080_420.pwtv",
	L"1920x1080_444.pwtv",
	L"3840x2160_420.pwtv",
	L"3840x2160_444.pwtv",
};

std::string run_tests()
{
	using namespace PyroWaveTest;
	std::string report = "=== PyroWave D3D12 decoder test ===\n";

	const std::wstring dir = std::wstring(Package::Current().InstalledLocation().Path()) + L"\\vectors\\";
	std::vector<TestVector> vectors;
	for (auto *file : vector_files)
	{
		TestVector vec;
		if (!read_test_vector(dir + file, vec))
		{
			appendf(report, "[FAIL] could not read vectors\\%ls\n", file);
			continue;
		}
		std::wstring w = file;
		w = w.substr(0, w.find_last_of(L'.'));
		for (wchar_t c : w)
			vec.name += char(c);
		vectors.push_back(std::move(vec));
	}

	Context ctx;
	if (!ctx.init(false, false))
		return report + "[FAIL] could not create a D3D12 device\n";

	bool pass = false;
	report += run_suite(ctx, vectors, 50, pass);
	return report;
}
}

struct App : implements<App, IFrameworkViewSource, IFrameworkView>
{
	IFrameworkView CreateView() { return *this; }

	void Initialize(CoreApplicationView const &view)
	{
		view.Activated({ this, &App::OnActivated });
	}

	void Load(hstring const &) {}
	void Uninitialize() {}
	void SetWindow(CoreWindow const &) {}

	void Run()
	{
		CoreWindow::GetForCurrentThread().Dispatcher().ProcessEvents(CoreProcessEventsOption::ProcessUntilQuit);
		if (worker.joinable())
			worker.join();
	}

	void OnActivated(CoreApplicationView const &, IActivatedEventArgs const &)
	{
		CoreWindow window = CoreWindow::GetForCurrentThread();
		window.Activate();
		if (started)
			return;
		started = true;

		const std::wstring path = std::wstring(ApplicationData::Current().LocalFolder().Path()) +
		                          L"\\pyrowave_d3d12_test.txt";
		std::ofstream(path, std::ios::binary) << "running\n";

		// Loading ~50 MB of vectors and decoding each 100 times takes a while; keep the
		// UI thread free so the shell does not consider the app hung.
		auto dispatcher = window.Dispatcher();
		worker = std::thread([dispatcher, path]() {
			_putenv("PYROWAVE_D3D12_PROFILE=1");
			std::string report = run_tests();
			OutputDebugStringA(report.c_str());
			std::ofstream(path, std::ios::binary) << report;
			dispatcher.RunAsync(CoreDispatcherPriority::Normal, [report]() {
				MessageDialog dialog(to_hstring(report), L"PyroWave D3D12 test");
				dialog.ShowAsync();
			});
		});
	}

	bool started = false;
	std::thread worker;
};

int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int)
{
	init_apartment();
	CoreApplication::Run(make<App>());
	return 0;
}
