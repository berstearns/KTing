case $1 in
 build)
 docker build -t kting-py312 . 
 ;;
 run)
 case $2 in
  dev)
  if [ -z "$3" ]; then
    echo "Error: Please specify a path"
    exit 1
  fi
  VPATH=$(realpath "$3") || exit 1
  echo "dev container"
  echo "Mounting: $VPATH"
  read -p "Press enter to continue..."
  docker run -it -v "$VPATH:/KTing" kting-py312 bash
  ;;
  *)
  docker run -it kting-py312 bash 
  ;;
 esac
 ;;
 *)
 echo "command not valid"
 ;;
esac
