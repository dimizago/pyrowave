// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Xbox (UWP) front-end for the D3D12 decoder test: replays every test vector packaged
// under vectors\ (dumped on a PC with pyrowave-d3d12-decode-compare --dump), shows
// the report in a dialog and writes it to LocalState\pyrowave_d3d12_test.txt, which
// Device Portal's File explorer can download.
//
// The package declares the hevcPlayback capability like the Moonlight client does: on
// Xbox that puts the app in the reduced "4K media app" resource tier regardless of the
// Dev Home app type, so the timings here are the ones the client gets. It decodes the
// way the client does (FP16 wavelet storage, R16_UNORM planes) and first characterises
// the GPU partition with a few micro-benchmarks.

#include "tier_bench.hpp"
#include "vector_test.hpp"

#include <algorithm>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <thread>

#include <dxgi1_4.h>
#include <expandedresources.h>

#include <winrt/Windows.ApplicationModel.Activation.h>
#include <winrt/Windows.ApplicationModel.Core.h>
#include <winrt/Windows.ApplicationModel.h>
#include <winrt/Windows.Foundation.Collections.h>
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
// GPU memory budget of the decoder's adapter: tells the resource tier apart (App ~1.2 GB,
// hevcPlayback media app ~2.8 GB, Game ~4.1 GB).
std::string memory_budget(ID3D12Device *device)
{
	com_ptr<IDXGIFactory4> factory;
	com_ptr<IDXGIAdapter3> adapter;
	DXGI_QUERY_VIDEO_MEMORY_INFO local = {};
	if (FAILED(CreateDXGIFactory2(0, __uuidof(IDXGIFactory4), factory.put_void())) ||
	    FAILED(factory->EnumAdapterByLuid(device->GetAdapterLuid(), __uuidof(IDXGIAdapter3), adapter.put_void())) ||
	    FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)))
		return "Memory budget: unavailable\n";
	char buf[96];
	snprintf(buf, sizeof(buf), "Memory budget: %llu MB (used %llu MB)\n", local.Budget >> 20, local.CurrentUsage >> 20);
	return buf;
}

// Game Mode as the system granted it (expandedresources.h, delay-loaded). On Xbox it is
// on for the Game app type or the expandedResources capability; these runs check what
// hevcPlayback does to it. Kept free of C++ objects for __try.
bool query_expanded_resources(BOOL &expanded, ULONG &cpus)
{
	__try
	{
		return SUCCEEDED(HasExpandedResources(&expanded)) &&
		       SUCCEEDED(GetExpandedResourceExclusiveCpuCount(&cpus));
	}
	__except (EXCEPTION_EXECUTE_HANDLER)
	{
		return false;
	}
}

// What this run was set up with, so reports from different capability / Dev Home app
// type combinations can't be mixed up: the capabilities the installed manifest declares
// and whether the system granted Game Mode resources. The app can't see the Dev Home
// app type itself, so note that when saving the report.
std::string run_setup()
{
	std::string report = "Declared capabilities:";
	std::ifstream manifest(std::wstring(Package::Current().InstalledLocation().Path()) + L"\\AppxManifest.xml");
	std::stringstream xml;
	xml << manifest.rdbuf();
	const std::string text = xml.str();
	const std::string key = "Capability Name=\"";
	for (size_t pos = text.find(key); pos != std::string::npos; pos = text.find(key, pos))
	{
		pos += key.size();
		report += " " + text.substr(pos, text.find('"', pos) - pos);
	}
	report += "\n";

	BOOL expanded = FALSE;
	ULONG cpus = 0;
	if (query_expanded_resources(expanded, cpus))
		PyroWaveTest::appendf(report, "Expanded resources (Game Mode): %s, exclusive CPU cores: %lu\n", expanded ? "yes" : "no", cpus);
	else
		report += "Expanded resources (Game Mode): unavailable\n";
	return report;
}

std::string run_tests()
{
	using namespace PyroWaveTest;
	std::string report = "=== PyroWave D3D12 decoder test ===\n";

	// Every .pwtv under vectors\, in name order.
	std::vector<std::wstring> files;
	const std::wstring dir = std::wstring(Package::Current().InstalledLocation().Path()) + L"\\vectors";
	for (auto const &file : StorageFolder::GetFolderFromPathAsync(dir).get().GetFilesAsync().get())
		if (file.FileType() == L".pwtv")
			files.push_back(std::wstring(file.Name()));
	std::sort(files.begin(), files.end());

	std::vector<TestVector> vectors;
	for (auto &file : files)
	{
		TestVector vec;
		if (!read_test_vector(dir + L"\\" + file, vec))
		{
			appendf(report, "[FAIL] could not read vectors\\%ls\n", file.c_str());
			continue;
		}
		std::wstring w = file.substr(0, file.find_last_of(L'.'));
		for (wchar_t c : w)
			vec.name += char(c);
		vectors.push_back(std::move(vec));
	}

	Context ctx;
	if (!ctx.init(false, false))
		return report + "[FAIL] could not create a D3D12 device\n";

	report += run_setup();
	report += memory_budget(ctx.device.get());
	report += run_tier_bench(ctx);

	g_output_format = DXGI_FORMAT_R16_UNORM;
	g_precisions = { 1 };
	// A/B: the translated (portable) kernels, then the Xbox ones Moonlight uses.
	for (const char *kernels : { "PYROWAVE_D3D12_XBOX_KERNELS=0", "PYROWAVE_D3D12_XBOX_KERNELS=1" })
	{
		_putenv(kernels);
		bool pass = false;
		appendf(report, "\n##### %s #####\n", kernels);
		report += run_suite(ctx, vectors, 60, pass);
	}
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

		// Each run also keeps its own copy, so redeploying for the next case can't lose a
		// report that hasn't been downloaded yet.
		wchar_t stamp[32];
		const time_t now = time(nullptr);
		wcsftime(stamp, 32, L"%Y%m%d_%H%M%S", localtime(&now));
		const std::wstring copy = std::wstring(ApplicationData::Current().LocalFolder().Path()) +
		                          L"\\pyrowave_d3d12_test_" + stamp + L".txt";

		// Loading ~100 MB of vectors and decoding each many times takes a while; keep
		// the UI thread free so the shell does not consider the app hung.
		auto dispatcher = window.Dispatcher();
		worker = std::thread([dispatcher, path, copy]() {
			_putenv("PYROWAVE_D3D12_PROFILE=1");
			std::string report = run_tests();
			OutputDebugStringA(report.c_str());
			std::ofstream(path, std::ios::binary) << report;
			std::ofstream(copy, std::ios::binary) << report;
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
