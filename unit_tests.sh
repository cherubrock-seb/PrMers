#!/bin/bash

mkdir -p logs

prime_exponents=(
    89 107 127 521 607 1279 2203 2281
    3217 4253 4423 9689 9941 11213 19937
    21701 23209 44497 86243 110503 132049
)

composite_exponents=(
    57 91 100 200 500 1001 4095 8191
)

echo "=== Prime exponents ==="
for p in "${prime_exponents[@]}"; do
    echo -n "Testing M$p... "
    ./prmers "$p" --noask -prp > "logs/ok_${p}.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Failed (see logs/ok_${p}.log)"
        exit 1
    else
        echo "✅"
    fi
done

echo ""
echo "=== Composite exponents ==="
for p in "${composite_exponents[@]}"; do
    echo -n "Testing M$p (composite)... "
    ./prmers "$p" --noask -ll > "logs/fail_${p}.log" 2>&1
    if [ $? -eq 0 ]; then
        echo "❌ Unexpected success"
        exit 1
    else
        echo "✅"
    fi
done

echo -e "\n🎉 All tests passed."
