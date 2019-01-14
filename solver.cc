namespace Solver{
  bool Solve(int puzzle[9][9]) {
    return SolveSpec (puzzle, 0, 0);
}

void Increment(int* row, int* col) {
  ++col;
  if (col == 9) {
    col = 0;
    ++row;
  }
}

bool SolveSpec(int puzzle[9][9], int row, int col) {
  if (puzzle[col][row] != 0) {
    Increment(&row, &col);
    if (row == 9)
      return true;
    if SolveSpec(puzzle, row, col)
     return true;
  }
  for (int nbr = 1; nbr < 10; ++nbr) {
  puzzle[col][row] = nbr;
   if (CheckRow(puzzle, row) && CheckCol(puzzle, col) && CheckBox(puzzle, row, col)) {
     Increment(&row, &col);
     if (row == 9)
       return true;
     if SolveSpec(puzzle, row, col)
      return true;
   }
   puzzle[col][row] = 0;
   return false;
 }
}

bool CheckRow(int puzzle[9][9], int row) {
  for (int i = 0; i < 9; ++i) {
    if (puzzle[i][row] == 0)
      continue;
    int current = puzzle[i][row];
    for (int j = i+1; j < 9; ++j)  {
      if (puzzle[j][row] == current)
        return false;
    }
  }
  return true;
}

bool CheckCol(int puzzle[9][9], int col) {
  for (int i = 0; i < 9; ++i) {
    if (puzzle[col][i] == 0)
      continue;
    int current = puzzle[col][i];
    for (int j = i+1; j < 9; ++j)  {
      if (puzzle[col][j] == current)
        return false;
    }
  }
  return true;
}

bool CheckBox(int puzzle[9][9], int row, int col) {
  while (row % 3 != 0)
    --row;
  while (col % 3 != 0)
    --col;
    int values[9];
    int count = 0;
  for (int i = row; i < row+3; ++i)
    for (int j = col; j < col+3; ++j) {
      values[count] = puzzle[i][j];
      ++count;
    }

    for (int i = 0; i < 9; ++i) {
      if (values[i] == 0)
        continue;
      int current = values[i];
      for (int j = i+1; j < 9; ++j)  {
        if (values[j] == current)
          return false;
      }
    }

  return true;
}

}
