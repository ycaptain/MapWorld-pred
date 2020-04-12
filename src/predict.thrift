namespace py mapworld

exception Exception {
  1: string message
}

struct InitRequest {
  1: string config_path
  2: string model_path
  3: optional string fr_addr
  4: optional i32 fr_port
}

struct PredRequest {
  1: list<string> imgs_path
}

service MapWorldService {
    void initialize(1: InitRequest req) throws (1: Exception e)
    void deinit()
    void doPred(1: PredRequest req) throws (1: Exception e)
}