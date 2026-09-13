#include <iostream>
#include <map>
#include <queue>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>


// The side of the trade
enum class Side {
    BUY,
    SELL
};

// Represents a single order in the market
struct Order {
    uint64_t id;
    Side side;
    double price;
    uint32_t quantity;
    uint64_t timestamp;
};


class OrderBook {
private:
    // Bids: Buyers want to buy as low as possible, but the exchange executes the HIGHEST bid first.
    // std::greater<double> automatically sorts the map in descending order.
    std::map<double, std::queue<Order>, std::greater<double>> bids;
    
    // Asks: Sellers want to sell as high as possible, but the exchange executes the LOWEST ask first.
    // std::less<double> automatically sorts the map in ascending order.
    std::map<double, std::queue<Order>, std::less<double>> asks;
    
    uint64_t next_order_id = 1;

    // Helper to get the current microsecond timestamp
    uint64_t getCurrentTime() {
        auto now = std::chrono::system_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    }

public:
    
    // Add a new order to the book and attempt to match it instantly
    void addOrder(Side side, double price, uint32_t quantity) {
        Order newOrder = {next_order_id++, side, price, quantity, getCurrentTime()};
        
        std::cout << "[NEW ORDER] ID: " << newOrder.id 
                  << " | Side: " << (side == Side::BUY ? "BUY " : "SELL") 
                  << " | Qty: " << quantity << " | Price: $" << price << "\n";

        if (side == Side::BUY) {
            matchBuyOrder(newOrder);
        } else {
            matchSellOrder(newOrder);
        }
    }

private:
    
    void matchBuyOrder(Order& buyOrder) {
        // While we have quantity to fill, and there are sellers, and our buy price is >= the lowest sell price
        while (buyOrder.quantity > 0 && !asks.empty() && buyOrder.price >= asks.begin()->first) {
            auto& bestAskQueue = asks.begin()->second;
            Order& bestAsk = bestAskQueue.front(); // Time priority: first in, first out

            // Calculate how much we can fill
            uint32_t fillQuantity = std::min(buyOrder.quantity, bestAsk.quantity);
            
            std::cout << "   --> [TRADE EXECUTED] " << fillQuantity << " shares @ $" 
                      << bestAsk.price << " (Matched Buy ID " << buyOrder.id 
                      << " with Sell ID " << bestAsk.id << ")\n";

            // Update quantities
            buyOrder.quantity -= fillQuantity;
            bestAsk.quantity -= fillQuantity;

            // If the resting sell order is completely filled, remove it from the queue
            if (bestAsk.quantity == 0) {
                bestAskQueue.pop();
            }

            // If the price level has no more orders, remove the price level from the map
            if (bestAskQueue.empty()) {
                asks.erase(asks.begin());
            }
        }

        // If the buy order still has quantity left, add it to the resting Bids book
        if (buyOrder.quantity > 0) {
            bids[buyOrder.price].push(buyOrder);
            std::cout << "   --> [RESTING] Added remaining " << buyOrder.quantity 
                      << " shares to Bids @ $" << buyOrder.price << "\n";
        }
    }

    
    void matchSellOrder(Order& sellOrder) {
        // While we have quantity to fill, and there are buyers, and our sell price is <= the highest buy price
        while (sellOrder.quantity > 0 && !bids.empty() && sellOrder.price <= bids.begin()->first) {
            auto& bestBidQueue = bids.begin()->second;
            Order& bestBid = bestBidQueue.front();

            // Calculate how much we can fill
            uint32_t fillQuantity = std::min(sellOrder.quantity, bestBid.quantity);
            
            std::cout << "   --> [TRADE EXECUTED] " << fillQuantity << " shares @ $" 
                      << bestBid.price << " (Matched Sell ID " << sellOrder.id 
                      << " with Buy ID " << bestBid.id << ")\n";

            // Update quantities
            sellOrder.quantity -= fillQuantity;
            bestBid.quantity -= fillQuantity;

            // Clean up filled orders and empty price levels
            if (bestBid.quantity == 0) {
                bestBidQueue.pop();
            }
            if (bestBidQueue.empty()) {
                bids.erase(bids.begin());
            }
        }

        // If the sell order still has quantity left, add it to the resting Asks book
        if (sellOrder.quantity > 0) {
            asks[sellOrder.price].push(sellOrder);
            std::cout << "   --> [RESTING] Added remaining " << sellOrder.quantity 
                      << " shares to Asks @ $" << sellOrder.price << "\n";
        }
    }

public:
    
    // Print the current state of the Limit Order Book (LOB)
    void printBook() {
        std::cout << "\n========================================\n";
        std::cout << "          LIMIT ORDER BOOK (LOB)        \n";
        std::cout << "========================================\n";
        
        // Print Asks (Sellers) - highest price at the top, lowest near the spread
        std::cout << "--- ASKS (Sellers) ---\n";
        for (auto it = asks.rbegin(); it != asks.rend(); ++it) {
            uint32_t totalVolume = 0;
            // Iterate through a copy of the queue to calculate total volume at this price
            std::queue<Order> tempQueue = it->second;
            while (!tempQueue.empty()) {
                totalVolume += tempQueue.front().quantity;
                tempQueue.pop();
            }
            std::cout << "   $" << std::fixed << std::setprecision(2) << it->first 
                      << "  |  Vol: " << totalVolume << "\n";
        }

        std::cout << "---------------------- [SPREAD] \n";

        // Print Bids (Buyers) - highest price near the spread, lowest at the bottom
        std::cout << "--- BIDS (Buyers) ---\n";
        for (auto it = bids.begin(); it != bids.end(); ++it) {
            uint32_t totalVolume = 0;
            std::queue<Order> tempQueue = it->second;
            while (!tempQueue.empty()) {
                totalVolume += tempQueue.front().quantity;
                tempQueue.pop();
            }
            std::cout << "   $" << std::fixed << std::setprecision(2) << it->first 
                      << "  |  Vol: " << totalVolume << "\n";
        }
        std::cout << "========================================\n\n";
    }
};


int main() {
    OrderBook engine;

    std::cout << "🚀 Initializing C++ Matching Engine...\n\n";

    // 1. Build the initial Order Book state (Resting Orders)
    engine.addOrder(Side::SELL, 102.50, 100);
    engine.addOrder(Side::SELL, 102.00, 50);
    engine.addOrder(Side::BUY, 100.50, 200);
    engine.addOrder(Side::BUY, 100.00, 150);

    engine.printBook();

    // 2. Introduce an aggressive BUY order that crosses the spread
    // This buyer wants 75 shares and is willing to pay up to $102.50. 
    // They should clear the 50 shares at $102.00, and take 25 shares from $102.50.
    engine.addOrder(Side::BUY, 102.50, 75);

    engine.printBook();

    // 3. Introduce a massive SELL dump
    // This seller wants to dump 250 shares at market/low price
    engine.addOrder(Side::SELL, 100.00, 250);

    engine.printBook();

    return 0;
}