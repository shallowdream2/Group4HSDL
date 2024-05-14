void generate_Group_louvain(std::unordered_map<int, std::vector<int>> &groups) {
  //运行python脚本，生成社区划分结果
  std::string cmd = "python3 louvain_clustering.py";
  int result = std::system(cmd.data());

  cpp_redis::client client;

  // Connect to Redis
  if (!client.is_connected()) {
    client.connect("127.0.0.1", 6379,
                   [](const std::string &host, std::size_t port,
                      cpp_redis::client::connect_state status) {
                     if (status == cpp_redis::client::connect_state::ok) {
                       std::cout << "Connected to redis" << std::endl;
                     } else {
                       // std::cout << "Failed to connect to redis" <<
                       // std::endl;
                     }
                   });
  }
  if (!client.is_connected()) {
    std::cerr << "Cannot connect to redis" << std::endl;
    return;
  }

  // Read community detection results from Redis
  client.hgetall("partition_result", [&groups](const cpp_redis::reply &reply) {
    if (reply.is_array()) {
      const auto &replies = reply.as_array();
      for (size_t i = 0; i < replies.size(); i += 2) {
        int node = std::stoi(replies[i].as_string());
        int community = std::stoi(replies[i + 1].as_string());
        groups[community].push_back(node);
      }
    }
  });

  // Synchronously commit the outstanding operations and wait for them to
  // complete
  client.sync_commit();

  printf("Redis operation completed\n");
}