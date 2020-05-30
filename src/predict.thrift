namespace py mapworld

exception Exception {
  1: string message
}

struct InitRequest {
  1: string config_path
  2: optional string fr_addr
  3: optional i32 fr_port
}

struct XY {
  1: double x
  2: double y
}

struct GeoMeta {
  1: XY origin // Upper left
  2: optional XY pixel_size
}

struct PredRequest {
  1: list<string> imgs_path
  2: list<GeoMeta> imgs_meta
  3: string model_name
  4: i8 n_gpu_use
  5: optional string tmp_opt_path
  6: optional double prescale
  7: optional i8 batch_size
}

struct Response {
  1: i8 code
  2: optional string msg
}

service MapWorldService {
    Response initialize(1: InitRequest req) throws (1: Exception e)
    Response deinit()
    Response doPred(1: PredRequest req) throws (1: Exception e)
    Response getTask()
}