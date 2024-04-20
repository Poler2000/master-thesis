module Logger

export log_info, log_debug, LOG_TO_CONSOLE, LOGFILE

ENABLE_DEBUG_LOGGING = false
LOG_TO_CONSOLE = true
LOGFILE = Nothing

function log_info(msg::String)
    if LOG_TO_CONSOLE
        println(msg)
    end

    if LOGFILE != Nothing
        open(LOGFILE, "a") do file
            write(file, msg)
        end
    end
end

function log_debug(msg::String)
    if !ENABLE_DEBUG_LOGGING
       return 
    end

    if LOG_TO_CONSOLE
        println(msg)
    end

    if LOGFILE != Nothing
        open(LOGFILE, "a") do file
            write(file, msg)
        end
    end
end

end