HOSTS=$1

echo "running dummy python processes ... ";
hudl -h $HOSTS "$PY dummy.py";
