/** Custom errors for the Samhati SDK. */

export class SamhatiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SamhatiError";
  }
}

export class APIError extends SamhatiError {
  public readonly statusCode: number;
  public readonly body?: unknown;

  constructor(message: string, statusCode: number, body?: unknown) {
    super(message);
    this.name = "APIError";
    this.statusCode = statusCode;
    this.body = body;
  }
}

export class AuthenticationError extends APIError {
  constructor(message: string = "Invalid or missing API key") {
    super(message, 401);
    this.name = "AuthenticationError";
  }
}

export class RateLimitError extends APIError {
  constructor(message: string = "Rate limit exceeded") {
    super(message, 429);
    this.name = "RateLimitError";
  }
}

export class TimeoutError extends SamhatiError {
  constructor(message: string = "Request timed out") {
    super(message);
    this.name = "TimeoutError";
  }
}

export class ConnectionError extends SamhatiError {
  constructor(message: string = "Failed to connect to Samhati node") {
    super(message);
    this.name = "ConnectionError";
  }
}
