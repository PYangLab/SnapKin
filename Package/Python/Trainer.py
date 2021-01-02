from SnapKin import run_snapkin
import sys

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Missing arguments file.')
    elif len(sys.argv) > 2:
        print('Too many arguments. Only one expected.')
    else:
        args_fp = sys.argv[1]
        try:
            # Load file
            with open(args_fp,'r') as arg_file:
                lines = arg_file.readlines()

            # Check args
            args_dict = {}
            for line in lines:
                arg_line = line.strip().split('=')
                if len(arg_line) != 2:
                    print('Error in argument file at {}'.format(line.strip()))
                    exit()
                args_dict[arg_line[0]] = arg_line[1]

            # Run SnapKin
            run_snapkin(args_dict)
        except FileNotFoundError:
            print('Arguments file not found. Please check that {} exists.'.format(args_fp))



