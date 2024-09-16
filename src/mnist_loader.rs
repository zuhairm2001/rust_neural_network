use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

const TRAIN_IMAGES: &str = "data/train-images.idx3-ubyte";
const TRAIN_LABELS: &str = "data/train-labels.idx1-ubyte";
const TEST_IMAGES: &str = "data/t10k-images.idx3-ubyte";
const TEST_LABELS: &str = "data/t10k-labels.idx1-ubyte";

pub struct MnistData {
    pub images: Vec<Vec<u8>>,
    pub labels: Vec<u8>,
}

pub fn load_mnist(images_file: &str, labels_file: &str) -> io::Result<MnistData> {
    let images = read_images(images_file)?;
    let labels = read_labels(labels_file)?;
    Ok(MnistData { images, labels })
}

fn read_images(filename: &str) -> io::Result<Vec<Vec<u8>>> {
    let file = File::open(filename).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to open image file '{}': {}", filename, e),
        )
    })?;
    let mut reader = BufReader::new(file);

    let magic_number = reader.read_u32::<BigEndian>()?;
    if magic_number != 2051 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid magic number for images file",
        ));
    }

    let num_images = reader.read_u32::<BigEndian>()? as usize;
    let num_rows = reader.read_u32::<BigEndian>()? as usize;
    let num_cols = reader.read_u32::<BigEndian>()? as usize;

    let mut images = Vec::with_capacity(num_images);
    let mut buffer = vec![0u8; num_rows * num_cols];

    for _ in 0..num_images {
        reader.read_exact(&mut buffer)?;
        images.push(buffer.clone());
    }

    Ok(images)
}

fn read_labels(filename: &str) -> io::Result<Vec<u8>> {
    let file = File::open(filename).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to open label file '{}': {}", filename, e),
        )
    })?;
    let mut reader = BufReader::new(file);

    let magic_number = reader.read_u32::<BigEndian>()?;
    if magic_number != 2049 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid magic number for labels file",
        ));
    }

    let num_labels = reader.read_u32::<BigEndian>()? as usize;

    let mut labels = Vec::with_capacity(num_labels);
    reader.read_to_end(&mut labels)?;

    Ok(labels)
}

pub fn load_training_data() -> io::Result<MnistData> {
    load_mnist(TRAIN_IMAGES, TRAIN_LABELS)
}

pub fn load_test_data() -> io::Result<MnistData> {
    load_mnist(TEST_IMAGES, TEST_LABELS)
}

// Helper function to print raw image data if needed
pub fn print_raw_image(image: &[u8]) {
    println!("Raw image data ({}x{} pixels):", 28, 28);
    println!("Vec<u8> = [");
    for (i, &pixel) in image.iter().enumerate() {
        print!("{:3}", pixel);
        if i % 28 == 27 {
            println!(",");
        } else {
            print!(", ");
        }
    }
    println!("]");
}
