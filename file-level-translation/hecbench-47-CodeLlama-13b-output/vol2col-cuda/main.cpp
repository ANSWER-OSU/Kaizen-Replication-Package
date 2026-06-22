#include<iostream>
#include<vector>
#include<thread>
#include<cassert>
#include<cmath>
#include<algorithm>
#include<memory>

template <typename T>
class scoped_allocator {
public:
    scoped_allocator() = default;

    scoped_allocator(std::size_t num_items_per_block)
        : m_num_items_per_block(num_items_per_block)
    {
    }

    T* allocate(std::size_t num_items)
    {
        assert(num_items <= m_num_items_per_block);
        return new T[num_items];
    }

    void deallocate(T* pointer, std::size_t num_items)
    {
        assert(num_items <= m_num_items_per_block);
        delete[] pointer;
    }

private:
    std::size_t m_num_items_per_block = 0;
};

template <typename T>
constexpr T square(T value) noexcept
{
    return value * value;
}

template <typename T>
float average_execution_time(
    const std::vector<T>& values,
    unsigned int num_samples = 10)
{
    long sum = 0;
    for (unsigned int i = 0; i < num_samples; ++i) {
        sum += values[i];
    }
    return static_cast<float>(sum) / num_samples;
}

template <typename T>
void vol2col_reference(
    const std::vector<T>& data_vol,
    unsigned int channels,
    unsigned int depth,
    unsigned int height,
    unsigned int width,
    unsigned int kernel_t,
    unsigned int kernel_h,
    unsigned int kernel_w,
    int pad_t,
    int pad_h,
    int pad_w,
    int stride_t,
    int stride_h,
    int stride_w,
    int dilation_t,
    int dilation_h,
    int dilation_w,
    std::vector<T>& data_col)
{
    unsigned int depth_col = (depth + 2 * pad_t - kernel_t) / stride_t + 1;
    unsigned int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    unsigned int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    for (unsigned int i = 0; i < kernel_h*kernel_w*kernel_t; ++i) {
        int x = i % kernel_w;
        int y = i / kernel_w % kernel_h;
        int z = i / kernel_h / kernel_w;

        T* c = &data_col[i * channels * depth_col * height_col * width_col];
        const T* v = &data_vol[((i % channels) * depth + z) * height + y] * width + x;
    }
}

template <typename T>
void col2vol_reference(
    const std::vector<T>& data_col,
    unsigned int channels,
    unsigned int depth,
    unsigned int height,
    unsigned int width,
    unsigned int kernel_t,
    unsigned int kernel_h,
    unsigned int kernel_w,
    int pad_t,
    int pad_h,
    int pad_w,
    int stride_t,
    int stride_h,
    int stride_w,
    int dilation_t,
    int dilation_h,
    int dilation_w,
    std::vector<T>& data_vol)
{
    unsigned int depth_col = (depth + 2 * pad_t - kernel_t) / stride_t + 1;
    unsigned int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    unsigned int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    for (unsigned int i = 0; i < channels * depth_col * height_col * width_col; ++i) {
        int x = i % width_col;
        int y = i / width_col % height_col;
        int z = i / height_col / width_col % depth_col;

        T* v = &data_col[((i % channels) * kernel_t + z) * kernel_w * kernel_h + y] * width_col + x;
        const T* c = &data_col[i * channels * depth_col * height_col * width_col];
    }
}

int get_blocks(std::size_t n)
{
	return n > 0 ? n : 1;
}

template<class T>
class threads_environment final {
public:
	template<typename F>
	threads_environment(unsigned int num_threads, F&& task, int num_blocks)
		: m_threads(num_threads)
	{
		for (auto& thread : m_threads) {
			thread = std::thread(std::forward<F>(task), num_blocks / num_threads);
		}
	}

	threads_environment(threads_environment&& other) noexcept
		: m_threads(std::move(other.m_threads))
	{
	}

	~threads_environment()
	{
		join();
	}

	threads_environment(const threads_environment&) = delete;
	threads_environment& operator=(const threads_environment&) = delete;
	threads_environment& operator=(threads_environment&&) = delete;

	void join()
	{
		for (auto& thread : m_threads) {
			std::move(thread).join();
		}
	}

private:
	threads_environment(const threads_environment&);
	threads_environment& operator=(const threads_environment&);
	threads_environment& operator=(threads_environment&&);

	std::vector<std::thread> m_threads;
};

template <typename T>
void eval(
	const int repeat,
	const int channels,
	const int depth,
	const int height,
	const int width,
	const int depth_col,
	const int height_col,
	const int width_col,
	const int ksize_t,
	const int ksize_h,
	const int ksize_w,
	const int pad_t,
	const int pad_h,
	const int pad_w,
	const int stride_t,
	const int stride_h,
	const int stride_w,
	const int dilation_t,
	const int dilation_h,
	const int dilation_w)
{
	std::shared_ptr<typename std::vector<T> > h_data_vol(new std::vector<T>(channels*depth*height*width));
	std::shared_ptr<typename std::vector<T> > h_data_col(new std::vector<T>(channels*ksize_t*ksize_h*ksize_w*
		depth_col*height_col*width_col));

	std::shared_ptr<typename std::vector<T> > h_data_vol_ref(new std::vector<T>(channels*depth*height*width));
	std::shared_ptr<typename std::vector<T> > h_data_col_ref(new std::vector<T>(channels*ksize_t*ksize_h*ksize_w*
		depth_col*height_col*width_col));

	scoped_allocator<T> allocator(1024);

	vol2col_reference<T>(*h_data_vol.get(), channels, depth, height, width, ksize_t, ksize_h, ksize_w,
		pad_t, pad_h, pad_w, stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w, *h_data_col_ref.get());


	std::vector<std::vector<T>> vol2col_timings(repeat);
	std::vector<std::vector<T>> col2vol_timings(repeat);

	for (int i = 0; i < repeat; i++) {
		threads_environment<T> threads(std::thread::hardware_concurrency(), [&](int num_blocks) {
			vol2col_kernel(num_blocks, allocator, *h_data_vol.get(), depth, height, width,
				ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w,
				stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w,
				depth_col, height_col, width_col, *h_data_col.get());
		}, get_blocks(channels*depth_col*height_col*width_col));

		vol2col_timings[i] = threads.join();
	}

	for (int i = 0; i < repeat; i++) {
		threads_environment<T> threads(std::thread::hardware_concurrency(), [&](int num_blocks) {
			col2vol_kernel(num_blocks, allocator, *h_data_col_ref.get(), depth, height, width,
				ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w,
				stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w,
				depth_col, height_col, width_col, *h_data_vol_ref.get());
		}, get_blocks(channels*depth_col*height_col*width_col));

		col2vol_timings[i] = threads.join();
	}
}

template<typename T>
void vol2col_kernel(unsigned int n, scoped_allocator<T>& allocator, const std::vector<T>& h_data_vol,
	unsigned int depth, unsigned int height, unsigned int width, unsigned int ksize_t,
	unsigned int ksize_h, unsigned int ksize_w, int pad_t, int pad_h, int pad_w, int stride_t,
	int stride_h, int stride_w, int dilation_t, int dilation_h, int dilation_w,
	unsigned int depth_col, unsigned int height_col, unsigned int width_col, std::vector<T>& h_data_col)
{
	auto start = std::chrono::steady_clock::now();

	for (unsigned int nblock = 0; nblock < n; ++nblock) {
		unsigned int w_out = nblock % width_col;
		unsigned int idx = nblock / width_col;
		unsigned int h_out = idx % height_col;
		idx = idx / height_col;
		unsigned int t_out = idx % depth_col;
		unsigned int channel_in = idx / depth_col;
		unsigned int channel_out = channel_in * ksize_t * ksize_h * ksize_w;
		unsigned int t_in = t_out*stride_t - pad_t;
		unsigned int h_in = h_out * stride_h - pad_h;
		unsigned int w_in = w_out * stride_w - pad_w;
		const T* v = &h_data_vol[((channel_in*depth + t_in)*height + h_in)*width + w_in];
		T* c = &h_data_col[((channel_out*depth_col + t_out)*height_col + h_out)*width_col + w_out];
		for (unsigned int i = 0; i < ksize_t; ++i) {
			for (unsigned int j = 0; j < ksize_h; ++j) {
				for (unsigned int k = 0; k < ksize_w; ++k) {
					unsigned int t = t_in + i*dilation_t;
					unsigned int h = h_in + j*dilation_h;
					unsigned int w = w_in + k*dilation_w;
					if (t >= 0 && h >= 0 && w >= 0
						&& t < depth && h < height && w < width) {
						unsigned int index =
							(i*dilation_t*height*width +
								j*dilation_h*width +
								k*dilation_w);
						*c = v[index];
					}
					c += depth_col*height_col*width_col;
				}
			}
		}
	}
}

template<typename T>
void col2vol_kernel(unsigned int n, scoped_allocator<T>& allocator, const std::vector<T>& h_data_col,
	unsigned int depth, unsigned int height, unsigned int width, unsigned int kernel_t,
	unsigned int kernel_h, unsigned int kernel_w, int pad_t, int pad_h, int pad_w, int stride_t,
	int stride_h, int stride_w, int dilation_t, int dilation_h, int dilation_w,
	unsigned int depth_col, unsigned int height_col, unsigned int width_col, std::vector<T>& h_data_vol)
{
	auto start = std::chrono::steady_clock::now();

	for (unsigned int index = 0; index < n; ++index) {
		int w_im = index % width + pad_w;
		int index1 = (index / width) % height + pad_h;
		int t_im = (index / width / height) % depth + pad_t;
		int c_im = index / (width * height * depth);
		int kernel_extent_w = (kernel_w - 1)*dilation_w + 1;
		int kernel_extent_h = (kernel_h - 1)*dilation_h + 1;
		int kernel_extent_t = (kernel_t - 1)*dilation_t + 1;

		int w_col_start = (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
		int w_col_end = std::min(w_im / stride_w + 1, width_col);

		int h_col_start = (index1 < kernel_extent_h) ? 0 : (index1 - kernel_extent_h) / stride_h + 1;
		int h_col_end = std::min(index1 / stride_h + 1, height_col);

		int t_col_start = (t_im < kernel_extent_t) ? 0 : (t_im - kernel_extent_t) / stride_t + 1;
		int t_col_end = std::min(t_im / stride_t + 1, depth_col);

		T val = static_cast<T>(0);
		for (unsigned int t_col = t_col_start; t_col < t_col_end; t_col += 1) {
			for (unsigned int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
				for (unsigned int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
					int t_k = (t_im - t_col*stride_t);
					int h_k = (index1 - h_col*stride_h);
					int w_k = (w_im - w_col*stride_w);

					if (t_k % dilation_t == 0 && h_k % dilation_h == 0
						&& w_k % dilation_w == 0) {
						t_k /= dilation_t;
						h_k /= dilation_h;
						w_k /= dilation_w;

						int col_index = ((((c_im*kernel_t + t_k)*kernel_h + h_k)
							*kernel_w + w_k)*depth_col + t_col)
							*height_col*width_col + h_col;
						val += h_data_col[col_index];
						//math_utils::atomicDotAdd(val, h_data_col[col_index]);
					}
				}
			}
		}
		h_data_vol[index] = static_cast<T>(val);
	}
}

int main(int argc, char* argv[])
{
	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " <repeat>\n";
		return 1;
	}
	int argc1 = atoi(argv[1]);

	int channels = 4;
	int depth = 3;
	int height = 255;
	int width = 255;
	int pad_t = 1;
	int pad_h = 1;
	int pad_w = 1;
	int stride_t = 2;
	int stride_h = 2;
	int stride_w = 2;
	int dilation_t = 2;
	int dilation_h = 2;
	int dilation_w = 2;
	int depth_col = 3;
	int height_col = 255;
	int width_col = 255;

	for (int k = 1; k <= 9; k = k + 2) {
		std::cout << "\nkernel size: " << k << "\n";
		int ksize_t = k;
		int ksize_h = k;
		int ksize_w = k;

		std::vector<int> vol2col_timings(argc1);

		for (int i = 0; i < argc1; i++) {
			std::vector<int> threads_timings(6);
			threads_environment<vol2col_reference<T> > threads(6, [&](int num_blocks) {
				vol2col_kernel(num_blocks, vol2col_reference, channels, depth, height, width,
					ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w, stride_t, stride_h, stride_w,
					dilation_t, dilation_h, dilation_w, depth_col, height_col, width_col, *h_data_col.get());
			}, get_blocks(channels*depth_col*height_col*width_col));

			threads_timings[i] = threads.join();
		}
		std::cout << "Average execution time of vol2col kernel: " << average_execution_time(threads_timings) << " (us)\n";
	}
	return 0;
}