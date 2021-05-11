#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>

using namespace std;

class Cell {
public:
    Cell () {
        neighbors.resize(6);
    }

    void input() {
        cin >> cell_index >> richness;
        for (auto& n: neighbors) {
            cin >> n;
        }
    }
    int cell_index;
    int richness;
    vector<int> neighbors;
};

class Tree {
public:
    Tree () = default;
    Tree (int cell_index, int size, bool is_mine, bool is_dormant) :
        cell_index{cell_index}, size{size}, is_mine{is_mine}, is_dormant{is_dormant} {}
    void input() {
        cin >> cell_index >> size >> is_mine >> is_dormant;
    }
    int cell_index;
    int size;
    bool is_mine;
    bool is_dormant;
};


/*
void	pass_day(State &state)
{
	int orientaion = state.info.days % 6;
	for (int i = 0; i < state.grid.size(); i++)
	{
		if (state.grid[i].empty)
			continue;
		for (int j = 0; j < 3; j++)
		{
			s[orientaion].shadow[i][j];
		}
	}
}

bool	seed_action(State &s, Action a, int player, int info)
{
	(void)info;
	if (!s.grid[a.to].empty)
	{
		return true;
	}
	s.grid[a.to].empty = false;
	s.grid[a.to].player = player;
	s.grid[a.to].size = 0;
	s.grid[a.to].sleep = true;
	return false;
}

bool	complete_action(State &s, Action a, int player, int info)
{
	s.grid[a.from].empty = true;
	s.info.sun[player] += info;
	s.info.nutriments--;
	return false;
}

bool	grow_action(State &s, Action a, int player, int info)
{
	s.grid[a.from].size += 1;
	return false;
}

bool	wait_action(State &s, Action a, int player, int info)
{
	s.info.wait[player] = true;
	return false;
}
std::function<bool(State&, Action, int, int)> t[4] =
{seed_action, complete_action, grow_action, wait_action};
State   simulate_action(State& s, Action a)
{
    State   new_s(s);
	int nutriments = s.info.nutriments;
    if (!s.info.player)
	{
        new_s.a = a;
    } else
	{
        new_s.grid[a.from].sleep = true;
        new_s.info.sun[0] -= s.a.cost;
        new_s.info.sun[1] -= a.cost;
		t[static_cast<int>(a.type)](s, s.a, 0, nutriments);
		t[static_cast<int>(a.type)](s, a, 1, nutriments);
		if (s.info.wait[0] and s.info.wait[1])
		{
		}
	}
	new_s.info.player = !s.info.player;
}

void	pass_day(State &state)
{
	int orientaion = state.info.days % 6;
	for (int i = 0; i < state.grid.size(); i++)
	{
		if (state.grid[i].empty)
			continue;
		for (int j = 0; j < 3; j++)
		{
			s[orientaion].shadow[i][j];
		}
	}
}

bool	seed_action(State &s, Action a, int player, int info)
{
	(void)info;
	if (!s.grid[a.to].empty)
	{
		return true;
	}
	s.grid[a.to].empty = false;
	s.grid[a.to].player = player;
	s.grid[a.to].size = 0;
	s.grid[a.to].sleep = true;
	return false;
}

bool	complete_action(State &s, Action a, int player, int info)
{
	s.grid[a.from].empty = true;
	s.info.sun[player] += info;
	s.info.nutriments--;
	return false;
}

bool	grow_action(State &s, Action a, int player, int info)
{
	s.grid[a.from].size += 1;
	return false;
}

bool	wait_action(State &s, Action a, int player, int info)
{
	s.info.wait[player] = true;
	return false;
}

std::function<bool(State&, Action, int, int)> t[4] =
{seed_action, complete_action, grow_action, wait_action};
State   simulate_action(State& s, Action a)
{
    State   new_s(s);
	int nutriments = s.info.nutriments;
    if (!s.info.player)
	{
        new_s.a = a;
    } else
	{
        new_s.grid[a.from].sleep = true;
        new_s.info.sun[0] -= s.a.cost;
        new_s.info.sun[1] -= a.cost;
		t[static_cast<int>(a.type)](s, s.a, 0, nutriments);
		t[static_cast<int>(a.type)](s, a, 1, nutriments);
		if (s.info.wait[0] and s.info.wait[1])
		{
		}
	}
	new_s.info.player = !s.info.player;
}
*/

class Game {
private:
        int day = 0;
        int nutrients = 0;
        vector<Cell> board;
        vector<Tree> trees;
        vector<tuple<string,int,int>> possible_actions;
        int mySun;
        int oppSun;
        int score;
        int oppScore;
        int oppIsWaiting;

public:
    void inputInitData() {
        int numberOfCells;
        cin >> numberOfCells;
        for (int i = 0; i < numberOfCells; i++) {
            Cell cell;
            cell.input();
            board.push_back(cell);
        }
    }

    void inputInfo() {
        // input game info
        cin >> day;
        cin >> nutrients;
        cin >> mySun >> score;
        cin >> oppSun >> oppScore >> oppIsWaiting;

        // input trees info
        trees.clear();
        int numberOfTrees;
        cin >> numberOfTrees;
        for (int i = 0; i < numberOfTrees; i++) {
            Tree tree;
            tree.input();
            trees.push_back(tree);
        }

        // input possible actions
        possible_actions.clear();
        int numberOfPossibleMoves;
        cin >> numberOfPossibleMoves;
        for (int i = 0; i < numberOfPossibleMoves; i++) {
            string type;
            int arg1 = 0;
            int arg2 = 0;
            cin >> type;

            if (type == "WAIT") {
                possible_actions.push_back(make_tuple(type, arg1,arg2));
            } else if (type == "COMPLETE") {
                cin >> arg1;
                possible_actions.push_back(make_tuple(type, arg1,arg2));
            }
            else if (type == "GROW") {
                cin >> arg1;
                possible_actions.push_back(make_tuple(type, arg1,arg2));
            }
            else if (type == "SEED") {
                cin >> arg1;
                cin >> arg2;
                possible_actions.push_back(make_tuple(type, arg1,arg2));
            }
        }
    }

    // TODO: Please implement the algorithm in this function
    string compute_next_action() {
        string action = "WAIT"; // default

        return action;
    }
};

int main()
{
    Game game;
    game.inputInitData();

    while (true) {
        game.inputInfo();

        cout << game.compute_next_action() << endl;
    }
}
