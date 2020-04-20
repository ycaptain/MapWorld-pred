namespace py mapworld

exception Exception {
  1: string message
}

struct InitRequest {
  1: string config_path
  2: optional string fr_addr
  3: optional i32 fr_port
}

struct PredRequest {
  1: list<string> imgs_path
}

struct Response {
  1: i8 code
  2: optional string msg
}

service MapWorldService {
    Response initialize(1: InitRequest req) throws (1: Exception e)
    Response deinit()
    Response doPred(1: PredRequest req) throws (1: Exception e)
}