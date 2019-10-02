/**
 * @file main.cc
 * @brief The main file of the project
 */

#include <iostream>
#include <fstream>

#include "sad2D.h"

int main()
{
        deallog.depth_console(2);
        printf("Hello World!\n");
        #ifdef DEBUG
        sad2D::test();
        #endif
        return 0;
}