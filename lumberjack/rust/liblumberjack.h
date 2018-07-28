/* Binding from Rust to Cython  */

#include <cstdint>
#include <cstdlib>

// Define which data types can be requested or cast to.
// to serve as flags between Cython and Rust for data type conversions / creations
enum class DType {
  Float64,
  Int32,
};

// This enum is what Cython will use to read the data created from Rust
struct DataPtr {
  enum class Tag {
    Float64,
    Int32,
  };

  struct Float64_Body {
    double *data_ptr;
    uintptr_t len;
  };

  struct Int32_Body {
    int32_t *data_ptr;
    uintptr_t len;
  };

  Tag tag;
  union {
    Float64_Body float64;
    Int32_Body int32;
  };
};

// Container for individual item
struct DataElement {
  enum class Tag {
    Float64,
    Int32,
  };

  struct Float64_Body {
    double _0;
  };

  struct Int32_Body {
    int32_t _0;
  };

  Tag tag;
  union {
    Float64_Body float64;
    Int32_Body int32;
  };
};

extern "C" {

/* Binding from Rust to Cython  */

DataPtr add_by_scalar(DataPtr data_ptr, double scalar, bool inplace);

// Create Series from arange and pass back as DataPtr
DataPtr arange(int32_t start, int32_t stop, DType dtype);

//
DataPtr astype(DataPtr ptr, DType dtype);

DataPtr cumsum(DataPtr data_ptr);

// Reconstruct Series from DataPtr and let it fall out of scope to clear from memory.
void free_data(DataPtr data_ptr);

double mean(DataPtr data_ptr);

DataPtr multiply_by_scalar(DataPtr data_ptr, double scalar, bool inplace);

double series_map(uint8_t *source_ptr,
                  uint32_t source_len,
                  uint8_t *target_ptr,
                  uint32_t target_len,
                  uint8_t *func_ptr,
                  uint32_t func_len);

double sum(DataPtr data_ptr);

// Set an individual item on an existing vec
void verify(DataPtr data_ptr);

} // extern "C"

/* Binding from Rust to Cython  */
