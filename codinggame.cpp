#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

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

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/

int main()
{
    int numberOfCells; // 37
    cin >> numberOfCells; cin.ignore();
    for (int i = 0; i < numberOfCells; i++)
	{
        int index; // 0 is the center cell, the next cells spiral outwards
        int richness; // 0 if the cell is unusable, 1-3 for usable cells
        int neigh0; // the index of the neighbouring cell for each direction
        int neigh1;
        int neigh2;
        int neigh3;
        int neigh4;
        int neigh5;
        cin >> index >> richness >> neigh0 >> neigh1 >> neigh2 >> neigh3 >> neigh4 >> neigh5; cin.ignore();
    }

    // game loop
    while (1)
	{
        int day; // the game lasts 24 days: 0-23
        cin >> day; cin.ignore();
        int nutrients; // the base score you gain from the next COMPLETE action
        cin >> nutrients; cin.ignore();
        int sun; // your sun points
        int score; // your current score
        cin >> sun >> score; cin.ignore();
        int oppSun; // opponent's sun points
        int oppScore; // opponent's score
        bool oppIsWaiting; // whether your opponent is asleep until the next day
        cin >> oppSun >> oppScore >> oppIsWaiting; cin.ignore();
        int numberOfTrees; // the current amount of trees
        cin >> numberOfTrees; cin.ignore();
        for (int i = 0; i < numberOfTrees; i++)
		{
            int cellIndex; // location of this tree
            int size; // size of this tree: 0-3
            bool isMine; // 1 if this is your tree
            bool isDormant; // 1 if this tree is dormant
            cin >> cellIndex >> size >> isMine >> isDormant; cin.ignore();
        }
        int numberOfPossibleActions; // all legal actions
        cin >> numberOfPossibleActions; cin.ignore();
        for (int i = 0; i < numberOfPossibleActions; i++)
		{
            string possibleAction;
            getline(cin, possibleAction); // try printing something from here to start with
        }

        // Write an action using cout. DON'T FORGET THE "<< endl"
        // To debug: cerr << "Debug messages..." << endl;


        // GROW cellIdx | SEED sourceIdx targetIdx | COMPLETE cellIdx | WAIT <message>
        cout << "WAIT" << endl;
    }
}