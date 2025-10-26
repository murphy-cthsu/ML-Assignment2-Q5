file(GLOB_RECURSE SRCS *.cpp)
# Exclude atari and chess when ALE is not available
list(FILTER SRCS EXCLUDE REGEX ".*/atari/.*")
list(FILTER SRCS EXCLUDE REGEX ".*/chess/.*")

add_library(environment ${SRCS})
target_include_directories(
    environment PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    base
    atari
    chess
    go
    gomoku
    hex
    killallgo
    nogo
    othello
    rubiks
    tictactoe
    stochastic
    stochastic/puzzle2048
)
target_link_libraries(
    environment
    config
    utils
    # ale::ale-lib  # Only needed for Atari games
    ${OpenCV_LIBS}
)