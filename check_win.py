import argparse

def check_win(player1_sign, player2_sign) :
    if player1_sign == "rock" :

        if player2_sign == "rock" :
            return 0

        elif player2_sign == "paper" :
            return -1

        elif player2_sign == "cissors" :
            return 1

        else :
            print("Erreur : le signe du joueur 2 n'a pas été reconnu.")

    elif player1_sign == "paper" :
        
        if player2_sign == "rock" :
            return 1

        elif player2_sign == "paper" :
            return 0

        elif player2_sign == "cissors" :
            return -1

        else :
            print("Erreur : le signe du joueur 2 n'a pas été reconnu.")

    elif player1_sign == "cissors" :
        
        if player2_sign == "rock" :
             return -1

        elif player2_sign == "paper" :
            return 1

        elif player2_sign == "cissors" :
            return 0

        else :
            print("Erreur : le signe du joueur 2 n'a pas été reconnu.")
    else :
        print("Erreur : le signe du joueur 1 n'a pas été reconnu.")
    
    return 42

###

def main(p1, p2):
    print(check_win(p1, p2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p1", "--player1-sign", required=True, help="the sign played by the first player")
    ap.add_argument("-p2", "--player2-sign", required=True, help="the sign played by the second player")
    args = vars(ap.parse_args())
    main(args["player1_sign"], args["player2_sign"])