#include <iostream>
#include <map>
#include <queue>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <zmq.h> // ZeroMQ for Low-Latency IPC
#include <unistd.h> // For sleep simulation

// The side of the trade
enum class Side { BUY, SELL };

struct Order {
    uint64_t id;
    Side side;
    double price;
    uint32_t quantity;
    uint64_t timestamp;
};

class OrderBook {
private:
    std::map<double, std::queue<Order>, std::greater<double>> bids;
    std::map<double, std::queue<Order>, std::less<double>> asks;
    uint64_t next_order_id = 1;
    
    // ZeroMQ Context and Publisher Socket
    void* zmq_context;
    void* zmq_publisher;

    uint64_t getCurrentTime() {
        auto now = std::chrono::system_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    }

public:
    OrderBook() {
        // 1. Initialize ZeroMQ Publisher on port 5555
        std::cout << "[ZMQ] Starting ZeroMQ Publisher on tcp://*:5555...\n";
        zmq_context = zmq_ctx_new();
        zmq_publisher = zmq_socket(zmq_context, ZMQ_PUB);
        zmq_bind(zmq_publisher, "tcp://*:5555");
    }

    ~OrderBook() {
        zmq_close(zmq_publisher);
        zmq_ctx_destroy(zmq_context);
    }

    void addOrder(Side side, double price, uint32_t quantity) {
        Order newOrder = {next_order_id++, side, price, quantity, getCurrentTime()};
        
        std::cout << "[NEW ORDER] ID: " << newOrder.id 
                  << " | Side: " << (side == Side::BUY ? "BUY " : "SELL") 
                  << " | Qty: " << quantity << " | Price: $" << price << "\n";

        if (side == Side::BUY) matchBuyOrder(newOrder);
        else matchSellOrder(newOrder);
        
        // After every state change, instantly broadcast the new order book to Python
        broadcastBookState();
    }

private:
    void matchBuyOrder(Order& buyOrder) {
        while (buyOrder.quantity > 0 && !asks.empty() && buyOrder.price >= asks.begin()->first) {
            auto& bestAskQueue = asks.begin()->second;
            Order& bestAsk = bestAskQueue.front();
            uint32_t fillQuantity = std::min(buyOrder.quantity, bestAsk.quantity);
            
            std::cout << "   --> [TRADE] " << fillQuantity << " shares @ $" << bestAsk.price << "\n";

            buyOrder.quantity -= fillQuantity;
            bestAsk.quantity -= fillQuantity;

            if (bestAsk.quantity == 0) bestAskQueue.pop();
            if (bestAskQueue.empty()) asks.erase(asks.begin());
        }
        if (buyOrder.quantity > 0) bids[buyOrder.price].push(buyOrder);
    }

    void matchSellOrder(Order& sellOrder) {
        while (sellOrder.quantity > 0 && !bids.empty() && sellOrder.price <= bids.begin()->first) {
            auto& bestBidQueue = bids.begin()->second;
            Order& bestBid = bestBidQueue.front();
            uint32_t fillQuantity = std::min(sellOrder.quantity, bestBid.quantity);
            
            std::cout << "   --> [TRADE] " << fillQuantity << " shares @ $" << bestBid.price << "\n";

            sellOrder.quantity -= fillQuantity;
            bestBid.quantity -= fillQuantity;

            if (bestBid.quantity == 0) bestBidQueue.pop();
            if (bestBidQueue.empty()) bids.erase(bids.begin());
        }
        if (sellOrder.quantity > 0) asks[sellOrder.price].push(sellOrder);
    }

    // 2. The Low-Latency Broadcaster
    void broadcastBookState() {
        std::ostringstream payload;
        payload << "LOB|BIDS:";
        
        // Serialize Bids
        for (auto it = bids.begin(); it != bids.end(); ++it) {
            uint32_t totalVolume = 0;
            std::queue<Order> tempQueue = it->second;
            while (!tempQueue.empty()) { totalVolume += tempQueue.front().quantity; tempQueue.pop(); }
            payload << it->first << "@" << totalVolume << ",";
        }
        
        payload << "|ASKS:";
        
        // Serialize Asks
        for (auto it = asks.begin(); it != asks.end(); ++it) {
            uint32_t totalVolume = 0;
            std::queue<Order> tempQueue = it->second;
            while (!tempQueue.empty()) { totalVolume += tempQueue.front().quantity; tempQueue.pop(); }
            payload << it->first << "@" << totalVolume << ",";
        }

        std::string message = payload.str();
        
        // Fire into RAM via ZeroMQ socket
        zmq_send(zmq_publisher, message.c_str(), message.size(), 0);
    }
};

int main() {
    OrderBook engine;
    std::cout << "🚀 Engine Online & Broadcasting...\n\n";

    // Simulating a live market environment with slight delays
    engine.addOrder(Side::SELL, 102.50, 100); usleep(500000);
    engine.addOrder(Side::SELL, 102.00, 50);  usleep(500000);
    engine.addOrder(Side::BUY, 100.50, 200);  usleep(500000);
    engine.addOrder(Side::BUY, 100.00, 150);  usleep(1000000);

    // Aggressive cross
    engine.addOrder(Side::BUY, 102.50, 75);   usleep(1000000);

    // Massive sell dump
    engine.addOrder(Side::SELL, 100.00, 250); usleep(1000000);

    std::cout << "\n✅ Simulation Complete.\n";
    return 0;
}