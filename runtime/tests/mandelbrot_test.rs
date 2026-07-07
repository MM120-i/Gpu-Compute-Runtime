use runtime::context::GpuContext;
use std::fs;

fn write_ppm(path: &str, pixels: &[u32], width: u32, height: u32) {
    let mut contents: String = format!("P6\n{} {}\n255\n", width, height);

    for &iter in pixels {
        let (r, g, b) = if iter == 0 {
            (0u8, 0u8, 0u8)
        } 
        else {
            let t: f32 = iter as f32 / 200.0;
            let r: u8 = (t.min(1.0) * 255.0) as u8;
            let g: u8 = ((t * 0.6).min(1.0) * 255.0) as u8;
            let b: u8 = ((t * 0.4).min(1.0) * 255.0) as u8;
            (r, g, b)
        };

        contents.push(r as char);
        contents.push(g as char);
        contents.push(b as char);
    }

    fs::write(path, &contents).expect("write PPM");
}

#[test]
fn mandelbrot_center_is_in_set() {
    let mut ctx: GpuContext = GpuContext::new().expect("create GpuContext");
    let width: u32 = 256u32;
    let height: u32 = 256u32;
    let max_iters: u32 = 500u32;

    let pixels: Vec<u32> = runtime::demos::mandelbrot::render_gpu(
        &mut ctx, width, height, max_iters, -0.5, 0.0, 3.5,
    ).expect("render");

    let center: u32 = pixels[128 * width as usize + 128];

    assert_eq!(center, max_iters, "center of Mandelbrot set should not escape");
}

#[test]
fn mandelbrot_corner_escapes() {
    let mut ctx: GpuContext = GpuContext::new().expect("create GpuContext");
    let width: u32 = 128u32;
    let height: u32 = 128u32;
    let max_iters: u32 = 200u32;

    let pixels: Vec<u32> = runtime::demos::mandelbrot::render_gpu(
        &mut ctx, width, height, max_iters, 0.0, 0.0, 4.0,
    ).expect("render");

    let corner: u32 = pixels[0];
    
    assert!(corner < max_iters, "corner pixel should escape");
}

#[test]
fn mandelbrot_render_full_1080p() {
    let mut ctx: GpuContext = GpuContext::new().expect("create GpuContext");
    let width: u32 = 1920u32;
    let height: u32 = 1080u32;
    let max_iters: u32 = 1000u32;

    eprintln!("rendering {}x{} at {} iterations...", width, height, max_iters);

    let pixels: Vec<u32> = runtime::demos::mandelbrot::render_gpu(
        &mut ctx, width, height, max_iters, -0.5, 0.0, 3.5,
    ).expect("render");

    let path: &str = "mandelbrot.ppm";
    
    write_ppm(path, &pixels, width, height);
    eprintln!("saved to {}", path);
}
