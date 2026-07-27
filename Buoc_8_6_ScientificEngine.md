# Bước 8.6 — Scientific Engine

Scientific Engine độc lập Android UI, xử lý các phép toán khoa học, kiểm tra miền xác định, quản lý `DEG/RAD` và chuẩn hóa kết quả qua `NumberFormatter`.

## Phạm vi

```text
app/src/main/java/com/example/calculator/engine/
├── AngleMode.java
└── ScientificEngine.java
```

Chưa sửa `MainActivity.java`, `layout_scientific_calculator.xml` hoặc `HistoryManager.java`. Các nút Scientific sẽ được nối ở Bước 8.7.

## Acceptance chính

```text
√16          → 4
2³           → 8
2ʸ với y = 3 → 8
log10(100)   → 2
sin(30°)     → 0.5
```

---

## 8.6.1 — `AngleMode.java`

File:

```text
app/src/main/java/com/example/calculator/engine/AngleMode.java
```

```java
package com.example.calculator.engine;

public enum AngleMode {

    DEG,

    RAD
}
```

---

## 8.6.2 — `ScientificEngine.java`

File:

```text
app/src/main/java/com/example/calculator/engine/ScientificEngine.java
```

```java
package com.example.calculator.engine;

import com.example.calculator.utils.NumberFormatter;

import java.util.Locale;

/**
 * Xử lý toàn bộ business logic của Scientific Calculator.
 *
 * Class này:
 *
 * - Không phụ thuộc Android View
 * - Không biết Button hoặc TextView
 * - Không thao tác SharedPreferences
 * - Không tự cập nhật History
 *
 * MainActivity sẽ gửi dữ liệu vào engine và đọc ScientificResult.
 */
public class ScientificEngine {

    private static final double EPSILON =
            1e-12;

    /*
     * 170! vẫn nằm trong giới hạn double.
     *
     * 171! sẽ thành Infinity.
     */
    private static final int MAX_FACTORIAL_INPUT =
            170;


    private AngleMode angleMode;


    // =========================================================
    // CONSTRUCTOR
    // =========================================================

    public ScientificEngine() {

        /*
         * Scientific Calculator mặc định dùng Degree.
         */
        angleMode =
                AngleMode.DEG;
    }


    // =========================================================
    // ANGLE MODE
    // =========================================================

    public AngleMode getAngleMode() {

        return angleMode;
    }


    public void setAngleMode(
            AngleMode angleMode
    ) {

        if (angleMode == null) {
            return;
        }

        this.angleMode =
                angleMode;
    }


    /**
     * DEG → RAD
     * RAD → DEG
     */
    public AngleMode toggleAngleMode() {

        if (angleMode == AngleMode.DEG) {

            angleMode =
                    AngleMode.RAD;

        } else {

            angleMode =
                    AngleMode.DEG;
        }

        return angleMode;
    }


    // =========================================================
    // POWER
    // =========================================================

    /**
     * x²
     */
    public ScientificResult square(
            double value
    ) {

        String expression =
                NumberFormatter.format(value)
                        + "²";

        return compute(
                expression,
                () -> value * value
        );
    }


    /**
     * x³
     */
    public ScientificResult cube(
            double value
    ) {

        String expression =
                NumberFormatter.format(value)
                        + "³";

        return compute(
                expression,
                () -> value * value * value
        );
    }


    /**
     * xʸ
     */
    public ScientificResult power(
            double base,
            double exponent
    ) {

        String expression =
                NumberFormatter.format(base)
                        + " ^ "
                        + NumberFormatter.format(exponent);

        return compute(
                expression,
                () -> powerValue(
                        base,
                        exponent
                )
        );
    }


    // =========================================================
    // ROOT
    // =========================================================

    /**
     * √x
     */
    public ScientificResult squareRoot(
            double value
    ) {

        String expression =
                "√("
                        + NumberFormatter.format(value)
                        + ")";

        return compute(
                expression,
                () -> squareRootValue(value)
        );
    }


    /**
     * ∛x
     */
    public ScientificResult cubeRoot(
            double value
    ) {

        String expression =
                "∛("
                        + NumberFormatter.format(value)
                        + ")";

        return compute(
                expression,
                () -> Math.cbrt(value)
        );
    }


    /**
     * ʸ√x
     *
     * degree   = y
     * radicand = x
     */
    public ScientificResult yRoot(
            double degree,
            double radicand
    ) {

        String expression =
                NumberFormatter.format(degree)
                        + "√("
                        + NumberFormatter.format(radicand)
                        + ")";

        return compute(
                expression,
                () -> yRootValue(
                        degree,
                        radicand
                )
        );
    }


    // =========================================================
    // RECIPROCAL
    // =========================================================

    /**
     * 1/x
     */
    public ScientificResult reciprocal(
            double value
    ) {

        String expression =
                "1 ÷ "
                        + NumberFormatter.format(value);

        return compute(
                expression,
                () -> reciprocalValue(value)
        );
    }


    // =========================================================
    // FACTORIAL
    // =========================================================

    /**
     * x!
     *
     * Chỉ chấp nhận số nguyên, không âm và không lớn hơn 170.
     */
    public ScientificResult factorial(
            double value
    ) {

        String expression =
                NumberFormatter.format(value)
                        + "!";

        return compute(
                expression,
                () -> factorialValue(value)
        );
    }


    // =========================================================
    // TRIGONOMETRY
    // =========================================================

    public ScientificResult sin(
            double value
    ) {

        String expression =
                createAngleExpression(
                        "sin",
                        value
                );

        return compute(
                expression,
                () -> {

                    double radians =
                            toRadians(value);

                    return normalizeNearZero(
                            Math.sin(radians)
                    );
                }
        );
    }


    public ScientificResult cos(
            double value
    ) {

        String expression =
                createAngleExpression(
                        "cos",
                        value
                );

        return compute(
                expression,
                () -> {

                    double radians =
                            toRadians(value);

                    return normalizeNearZero(
                            Math.cos(radians)
                    );
                }
        );
    }


    public ScientificResult tan(
            double value
    ) {

        String expression =
                createAngleExpression(
                        "tan",
                        value
                );

        return compute(
                expression,
                () -> tangentValue(value)
        );
    }


    // =========================================================
    // LOGARITHM
    // =========================================================

    public ScientificResult naturalLog(
            double value
    ) {

        String expression =
                "ln("
                        + NumberFormatter.format(value)
                        + ")";

        return compute(
                expression,
                () -> naturalLogValue(value)
        );
    }


    public ScientificResult log10(
            double value
    ) {

        String expression =
                "log("
                        + NumberFormatter.format(value)
                        + ")";

        return compute(
                expression,
                () -> log10Value(value)
        );
    }


    // =========================================================
    // EXPONENTIAL
    // =========================================================

    public ScientificResult exponential(
            double value
    ) {

        String expression =
                "e^("
                        + NumberFormatter.format(value)
                        + ")";

        return compute(
                expression,
                () -> Math.exp(value)
        );
    }


    public ScientificResult tenPower(
            double value
    ) {

        String expression =
                "10^("
                        + NumberFormatter.format(value)
                        + ")";

        return compute(
                expression,
                () -> Math.pow(
                        10.0,
                        value
                )
        );
    }


    // =========================================================
    // CONSTANTS
    // =========================================================

    public ScientificResult constantPi() {

        return success(
                "π",
                Math.PI
        );
    }


    public ScientificResult constantE() {

        return success(
                "e",
                Math.E
        );
    }


    // =========================================================
    // EXPRESSION EVALUATOR
    // =========================================================

    /**
     * Hỗ trợ:
     *
     * - Số và decimal
     * - +, -, ×, ÷
     * - ^
     * - Ngoặc ()
     * - π, e
     * - Factorial postfix !
     * - sin(), cos(), tan()
     * - sqrt(), cbrt()
     * - ln(), log(), log10()
     * - exp()
     */
    public ScientificResult evaluateExpression(
            String expression
    ) {

        if (expression == null
                || expression.trim().isEmpty()) {

            return error(
                    "",
                    "Expression is empty."
            );
        }

        String cleanExpression =
                expression.trim();

        try {

            ExpressionParser parser =
                    new ExpressionParser(
                            cleanExpression
                    );

            double result =
                    parser.parse();

            return success(
                    cleanExpression,
                    result
            );

        } catch (IllegalArgumentException exception) {

            return error(
                    cleanExpression,
                    exception.getMessage()
            );
        }
    }


    // =========================================================
    // INTERNAL COMPUTATION
    // =========================================================

    private ScientificResult compute(
            String expression,
            Computation computation
    ) {

        try {

            double value =
                    computation.calculate();

            return success(
                    expression,
                    value
            );

        } catch (IllegalArgumentException exception) {

            return error(
                    expression,
                    exception.getMessage()
            );
        }
    }


    private double reciprocalValue(
            double value
    ) {

        if (Math.abs(value) < EPSILON) {

            throw new IllegalArgumentException(
                    "Cannot divide by zero."
            );
        }

        return 1.0 / value;
    }


    private double squareRootValue(
            double value
    ) {

        if (value < 0.0) {

            throw new IllegalArgumentException(
                    "Square root requires a non-negative value."
            );
        }

        return Math.sqrt(value);
    }


    private double factorialValue(
            double value
    ) {

        if (value < 0.0) {

            throw new IllegalArgumentException(
                    "Factorial requires a non-negative value."
            );
        }

        if (!isInteger(value)) {

            throw new IllegalArgumentException(
                    "Factorial requires an integer."
            );
        }

        if (value > MAX_FACTORIAL_INPUT) {

            throw new IllegalArgumentException(
                    "Factorial result is too large."
            );
        }

        int limit =
                (int) Math.rint(value);

        double result =
                1.0;

        for (int i = 2;
             i <= limit;
             i++) {

            result *= i;
        }

        return result;
    }


    private double tangentValue(
            double value
    ) {

        double radians =
                toRadians(value);

        double cosine =
                Math.cos(radians);

        if (Math.abs(cosine) < EPSILON) {

            throw new IllegalArgumentException(
                    "Tangent is undefined for this angle."
            );
        }

        return normalizeNearZero(
                Math.tan(radians)
        );
    }


    private double naturalLogValue(
            double value
    ) {

        if (value <= 0.0) {

            throw new IllegalArgumentException(
                    "Natural logarithm requires a positive value."
            );
        }

        return Math.log(value);
    }


    private double log10Value(
            double value
    ) {

        if (value <= 0.0) {

            throw new IllegalArgumentException(
                    "Logarithm requires a positive value."
            );
        }

        return Math.log10(value);
    }


    private double powerValue(
            double base,
            double exponent
    ) {

        double result =
                Math.pow(
                        base,
                        exponent
                );

        requireFinite(
                result,
                "Power result is outside the supported range."
        );

        return result;
    }


    private double yRootValue(
            double degree,
            double radicand
    ) {

        if (Math.abs(degree) < EPSILON) {

            throw new IllegalArgumentException(
                    "Root degree cannot be zero."
            );
        }

        if (radicand < 0.0) {

            if (!isInteger(degree)) {

                throw new IllegalArgumentException(
                        "A negative value requires an odd integer root."
                );
            }

            long integerDegree =
                    Math.round(degree);

            if (Math.abs(integerDegree) % 2 == 0) {

                throw new IllegalArgumentException(
                        "An even root of a negative value is not real."
                );
            }

            double result =
                    -Math.pow(
                            Math.abs(radicand),
                            1.0 / degree
                    );

            requireFinite(
                    result,
                    "Root result is outside the supported range."
            );

            return result;
        }

        double result =
                Math.pow(
                        radicand,
                        1.0 / degree
                );

        requireFinite(
                result,
                "Root result is outside the supported range."
        );

        return result;
    }


    // =========================================================
    // ANGLE HELPERS
    // =========================================================

    private double toRadians(
            double value
    ) {

        if (angleMode == AngleMode.DEG) {

            return Math.toRadians(value);
        }

        return value;
    }


    private String createAngleExpression(
            String functionName,
            double value
    ) {

        String formatted =
                NumberFormatter.format(value);

        if (angleMode == AngleMode.DEG) {

            return functionName
                    + "("
                    + formatted
                    + "°)";
        }

        return functionName
                + "("
                + formatted
                + ")";
    }


    // =========================================================
    // RESULT HELPERS
    // =========================================================

    private ScientificResult success(
            String expression,
            double value
    ) {

        if (!isFinite(value)) {

            return error(
                    expression,
                    "Result is outside the supported range."
            );
        }

        String formattedValue =
                NumberFormatter.format(value);

        if ("Error".equals(formattedValue)) {

            return error(
                    expression,
                    "Unable to calculate this value."
            );
        }

        return new ScientificResult(
                true,
                value,
                formattedValue,
                expression,
                null
        );
    }


    private ScientificResult error(
            String expression,
            String errorMessage
    ) {

        String safeMessage =
                errorMessage == null
                        || errorMessage.trim().isEmpty()

                        ? "Invalid operation."
                        : errorMessage;

        return new ScientificResult(
                false,
                Double.NaN,
                "Error",
                expression == null
                        ? ""
                        : expression,
                safeMessage
        );
    }


    private double normalizeNearZero(
            double value
    ) {

        if (Math.abs(value) < EPSILON) {

            return 0.0;
        }

        return value;
    }


    private boolean isInteger(
            double value
    ) {

        return Math.abs(
                value - Math.rint(value)
        ) < EPSILON;
    }


    private boolean isFinite(
            double value
    ) {

        return !Double.isNaN(value)
                && !Double.isInfinite(value);
    }


    private void requireFinite(
            double value,
            String message
    ) {

        if (!isFinite(value)) {

            throw new IllegalArgumentException(
                    message
            );
        }
    }


    // =========================================================
    // COMPUTATION CALLBACK
    // =========================================================

    private interface Computation {

        double calculate();
    }


    // =========================================================
    // RESULT MODEL
    // =========================================================

    public static final class ScientificResult {

        private final boolean success;

        private final double value;

        private final String formattedValue;

        private final String expression;

        private final String errorMessage;


        private ScientificResult(
                boolean success,
                double value,
                String formattedValue,
                String expression,
                String errorMessage
        ) {

            this.success =
                    success;

            this.value =
                    value;

            this.formattedValue =
                    formattedValue;

            this.expression =
                    expression;

            this.errorMessage =
                    errorMessage;
        }


        public boolean isSuccess() {

            return success;
        }


        public double getValue() {

            return value;
        }


        public String getFormattedValue() {

            return formattedValue;
        }


        public String getExpression() {

            return expression;
        }


        public String getErrorMessage() {

            return errorMessage;
        }
    }


    // =========================================================
    // EXPRESSION PARSER
    // =========================================================

    private final class ExpressionParser {

        private final String input;

        private int position;


        private ExpressionParser(
                String expression
        ) {

            input =
                    expression
                            .replace(
                                    '×',
                                    '*'
                            )
                            .replace(
                                    '÷',
                                    '/'
                            )
                            .replace(
                                    '−',
                                    '-'
                            );

            position =
                    0;
        }


        private double parse() {

            double value =
                    parseExpression();

            skipWhitespace();

            if (position != input.length()) {

                throw new IllegalArgumentException(
                        "Unexpected token at position "
                                + position
                                + "."
                );
            }

            requireFinite(
                    value,
                    "Expression result is outside the supported range."
            );

            return value;
        }


        private double parseExpression() {

            double value =
                    parseTerm();

            while (true) {

                skipWhitespace();

                if (match('+')) {

                    value +=
                            parseTerm();

                } else if (match('-')) {

                    value -=
                            parseTerm();

                } else {

                    return value;
                }
            }
        }


        private double parseTerm() {

            double value =
                    parseSignedPower();

            while (true) {

                skipWhitespace();

                if (match('*')) {

                    value *=
                            parseSignedPower();

                    requireFinite(
                            value,
                            "Multiplication result is outside the supported range."
                    );

                } else if (match('/')) {

                    double divisor =
                            parseSignedPower();

                    if (Math.abs(divisor) < EPSILON) {

                        throw new IllegalArgumentException(
                                "Cannot divide by zero."
                        );
                    }

                    value /=
                            divisor;

                } else {

                    return value;
                }
            }
        }


        private double parseSignedPower() {

            skipWhitespace();

            if (match('+')) {

                return parseSignedPower();
            }

            if (match('-')) {

                return -parseSignedPower();
            }

            return parsePower();
        }


        private double parsePower() {

            double base =
                    parsePostfix();

            skipWhitespace();

            if (match('^')) {

                double exponent =
                        parseSignedPower();

                return powerValue(
                        base,
                        exponent
                );
            }

            return base;
        }


        private double parsePostfix() {

            double value =
                    parsePrimary();

            while (true) {

                skipWhitespace();

                if (match('!')) {

                    value =
                            factorialValue(value);

                } else {

                    return value;
                }
            }
        }


        private double parsePrimary() {

            skipWhitespace();

            if (match('(')) {

                double value =
                        parseExpression();

                skipWhitespace();

                if (!match(')')) {

                    throw new IllegalArgumentException(
                            "Missing closing parenthesis."
                    );
                }

                return value;
            }

            if (peek('π')) {

                position++;

                return Math.PI;
            }

            char current =
                    currentCharacter();

            if (Character.isLetter(current)) {

                String identifier =
                        parseIdentifier()
                                .toLowerCase(
                                        Locale.US
                                );

                skipWhitespace();

                if ("e".equals(identifier)
                        && !peek('(')) {

                    return Math.E;
                }

                if ("pi".equals(identifier)
                        && !peek('(')) {

                    return Math.PI;
                }

                if (!match('(')) {

                    throw new IllegalArgumentException(
                            "Function "
                                    + identifier
                                    + " requires parentheses."
                    );
                }

                double argument =
                        parseExpression();

                skipWhitespace();

                if (!match(')')) {

                    throw new IllegalArgumentException(
                            "Missing closing parenthesis for "
                                    + identifier
                                    + "."
                    );
                }

                return applyFunction(
                        identifier,
                        argument
                );
            }

            return parseNumber();
        }


        private double applyFunction(
                String function,
                double argument
        ) {

            switch (function) {

                case "sin":

                    return normalizeNearZero(
                            Math.sin(
                                    toRadians(argument)
                            )
                    );

                case "cos":

                    return normalizeNearZero(
                            Math.cos(
                                    toRadians(argument)
                            )
                    );

                case "tan":

                    return tangentValue(argument);

                case "sqrt":

                    return squareRootValue(argument);

                case "cbrt":

                    return Math.cbrt(argument);

                case "ln":

                    return naturalLogValue(argument);

                case "log":

                case "log10":

                    return log10Value(argument);

                case "exp":

                    return Math.exp(argument);

                default:

                    throw new IllegalArgumentException(
                            "Unknown function: "
                                    + function
                                    + "."
                    );
            }
        }


        private double parseNumber() {

            skipWhitespace();

            int start =
                    position;

            boolean hasDigit =
                    false;

            while (position < input.length()
                    && Character.isDigit(
                            input.charAt(position)
                    )) {

                position++;

                hasDigit =
                        true;
            }

            if (peek('.')) {

                position++;

                while (position < input.length()
                        && Character.isDigit(
                                input.charAt(position)
                        )) {

                    position++;

                    hasDigit =
                            true;
                }
            }

            if (!hasDigit) {

                throw new IllegalArgumentException(
                        "Expected a number at position "
                                + position
                                + "."
                );
            }

            int exponentStart =
                    position;

            if (peek('E')
                    || peek('e')) {

                position++;

                if (peek('+')
                        || peek('-')) {

                    position++;
                }

                int exponentDigitStart =
                        position;

                while (position < input.length()
                        && Character.isDigit(
                                input.charAt(position)
                        )) {

                    position++;
                }

                if (exponentDigitStart == position) {

                    position =
                            exponentStart;
                }
            }

            String numberText =
                    input.substring(
                            start,
                            position
                    );

            try {

                return Double.parseDouble(
                        numberText
                );

            } catch (NumberFormatException exception) {

                throw new IllegalArgumentException(
                        "Invalid number: "
                                + numberText
                                + "."
                );
            }
        }


        private String parseIdentifier() {

            int start =
                    position;

            while (position < input.length()) {

                char character =
                        input.charAt(position);

                if (!Character.isLetterOrDigit(
                        character
                )) {

                    break;
                }

                position++;
            }

            return input.substring(
                    start,
                    position
            );
        }


        private void skipWhitespace() {

            while (position < input.length()
                    && Character.isWhitespace(
                            input.charAt(position)
                    )) {

                position++;
            }
        }


        private boolean match(
                char expected
        ) {

            skipWhitespace();

            if (position < input.length()
                    && input.charAt(position) == expected) {

                position++;

                return true;
            }

            return false;
        }


        private boolean peek(
                char expected
        ) {

            skipWhitespace();

            return position < input.length()
                    && input.charAt(position) == expected;
        }


        private char currentCharacter() {

            skipWhitespace();

            if (position >= input.length()) {

                return '\0';
            }

            return input.charAt(position);
        }
    }
}
```

---

## 8.6.3 — API sử dụng từ `MainActivity`

Ví dụ căn bậc hai:

```java
ScientificEngine.ScientificResult result =
        scientificEngine.squareRoot(
                16
        );
```

Kết quả:

```text
result.isSuccess()       → true
result.getExpression()   → √(16)
result.getFormattedValue() → 4
result.getValue()        → 4.0
```

Ví dụ lỗi:

```java
ScientificEngine.ScientificResult result =
        scientificEngine.squareRoot(
                -16
        );
```

```text
isSuccess()        → false
getFormattedValue() → Error
getErrorMessage()   → Square root requires a non-negative value.
```

UI chỉ cần đọc kết quả:

```java
tvSciExpression.setText(
        result.getExpression()
);

tvSciResult.setText(
        result.getFormattedValue()
);
```

---

## 8.6.4 — Ma trận kiểm thử logic

### Power và root

```text
square(5)           → 25
cube(2)             → 8
power(2, 3)         → 8
power(-4, 0.5)      → Error
squareRoot(16)      → 4
squareRoot(-1)      → Error
cubeRoot(-8)        → -2
yRoot(3, 8)         → 2
yRoot(2, -4)        → Error
```

### Reciprocal và factorial

```text
reciprocal(4)       → 0.25
reciprocal(0)       → Error
factorial(5)        → 120
factorial(0)        → 1
factorial(-1)       → Error
factorial(2.5)      → Error
factorial(171)      → Error
```

### DEG/RAD

```java
ScientificEngine engine =
        new ScientificEngine();

engine.sin(30);
// DEG mặc định → 0.5

engine.toggleAngleMode();
// RAD

engine.sin(
        Math.PI / 2
);
// → 1
```

### Logarithm và exponential

```text
log10(100)          → 2
naturalLog(e)       → 1
log10(0)            → Error
naturalLog(-1)      → Error
constantPi()        → khoảng 3.14159265359
constantE()         → khoảng 2.718281828459
exponential(1)      → khoảng 2.718281828459
tenPower(3)         → 1000
```

### Expression evaluator

```text
(2 + 3) × 4         → 20
2 + 3 × 4           → 14
2 ^ 3               → 8
5!                  → 120
sin(30) trong DEG   → 0.5
2 × π               → khoảng 6.28318530718
```

---

## 8.6.5 — Local unit test

Tạo file:

```text
app/src/test/java/com/example/calculator/engine/ScientificEngineTest.java
```

```java
package com.example.calculator.engine;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class ScientificEngineTest {

    private static final double DELTA =
            1e-10;


    @Test
    public void squareRootOf16_returns4() {

        ScientificEngine engine =
                new ScientificEngine();

        ScientificEngine.ScientificResult result =
                engine.squareRoot(16);

        assertTrue(
                result.isSuccess()
        );

        assertEquals(
                4.0,
                result.getValue(),
                DELTA
        );
    }


    @Test
    public void cubeOf2_returns8() {

        ScientificEngine engine =
                new ScientificEngine();

        ScientificEngine.ScientificResult result =
                engine.cube(2);

        assertEquals(
                8.0,
                result.getValue(),
                DELTA
        );
    }


    @Test
    public void power2To3_returns8() {

        ScientificEngine engine =
                new ScientificEngine();

        ScientificEngine.ScientificResult result =
                engine.power(
                        2,
                        3
                );

        assertEquals(
                8.0,
                result.getValue(),
                DELTA
        );
    }


    @Test
    public void sin30Degrees_returnsHalf() {

        ScientificEngine engine =
                new ScientificEngine();

        ScientificEngine.ScientificResult result =
                engine.sin(30);

        assertEquals(
                0.5,
                result.getValue(),
                DELTA
        );
    }


    @Test
    public void sinPiOver2Radians_returns1() {

        ScientificEngine engine =
                new ScientificEngine();

        engine.setAngleMode(
                AngleMode.RAD
        );

        ScientificEngine.ScientificResult result =
                engine.sin(
                        Math.PI / 2
                );

        assertEquals(
                1.0,
                result.getValue(),
                DELTA
        );
    }


    @Test
    public void log10Of100_returns2() {

        ScientificEngine engine =
                new ScientificEngine();

        ScientificEngine.ScientificResult result =
                engine.log10(100);

        assertEquals(
                2.0,
                result.getValue(),
                DELTA
        );
    }


    @Test
    public void factorial5_returns120() {

        ScientificEngine engine =
                new ScientificEngine();

        ScientificEngine.ScientificResult result =
                engine.factorial(5);

        assertEquals(
                120.0,
                result.getValue(),
                DELTA
        );
    }


    @Test
    public void squareRootOfNegative_returnsError() {

        ScientificEngine engine =
                new ScientificEngine();

        ScientificEngine.ScientificResult result =
                engine.squareRoot(-1);

        assertFalse(
                result.isSuccess()
        );

        assertEquals(
                "Error",
                result.getFormattedValue()
        );
    }


    @Test
    public void parenthesesRespectPrecedence() {

        ScientificEngine engine =
                new ScientificEngine();

        ScientificEngine.ScientificResult result =
                engine.evaluateExpression(
                        "(2 + 3) × 4"
                );

        assertEquals(
                20.0,
                result.getValue(),
                DELTA
        );
    }
}
```

Chạy:

```text
ScientificEngineTest
→ Run Tests
```

---

## 8.6.6 — Build checkpoint

```text
Build
→ Make Project
```

Checklist:

```text
[ ] AngleMode compile
[ ] ScientificEngine compile
[ ] Không import Android class trong ScientificEngine
[ ] ScientificResult compile
[ ] ExpressionParser compile
[ ] √16 → 4
[ ] 2³ → 8
[ ] 2^3 → 8
[ ] sin(30°) → 0.5
[ ] sin(π/2 rad) → 1
[ ] log10(100) → 2
[ ] 5! → 120
[ ] sqrt(-1) → Error
[ ] 1/0 → Error
[ ] tan(90°) → Error
[ ] (2 + 3) × 4 → 20
[ ] Basic Calculator không bị ảnh hưởng
[ ] History không bị ảnh hưởng
```

---

## Trạng thái sau Bước 8.6

```text
M3 — SCIENTIFIC CALCULATOR

[✓] 8.5 Scientific UI

[✓] 8.6 ScientificEngine
     ├── AngleMode DEG/RAD
     ├── x²
     ├── x³
     ├── xʸ
     ├── √x
     ├── ∛x
     ├── ʸ√x
     ├── 1/x
     ├── x!
     ├── sin/cos/tan
     ├── ln/log10
     ├── e/π
     ├── eˣ/10ˣ
     ├── parentheses evaluator
     ├── operator precedence
     └── domain/error validation

NEXT

[ ] 8.7 Scientific UI ↔ Engine
     ├── scientific engine instance
     ├── scientific display state
     ├── numeric keypad
     ├── unary buttons
     ├── xʸ pending operation
     ├── ʸ√x pending operation
     ├── parentheses input
     ├── DEG/RAD toggle
     ├── scientific equals
     └── save successful result to History
```
