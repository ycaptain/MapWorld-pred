namespace py mapworld

struct MsgHead {
  1: i32 from_client
  2: i32 msg_seq
  3: i64 timestamp
  4: string request_cmd
}

struct MsgBody {
  1: binary msg_content
}

struct Msg {
  1: MsgHead msg_head
  2: MsgBody msg_body
}

struct GetMessagesResponse {
  1: list<Msg> messages
}

struct SendMessageRequest {
  1: Msg messages
}

exception Exception {
  1: string message
}

service MapWorldService {
  /**
   * Initialize the service
   */
  void initialize() throws (1: Exception e)

  /**
   * Get the last few chat messages
   */
  GetMessagesResponse getMessages()
    throws (1: Exception e)

  /**
   * Send a message
   */
  void sendMessage(1: SendMessageRequest req) throws (1: Exception e)
}